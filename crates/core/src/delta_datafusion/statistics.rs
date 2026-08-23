//! Statistics requests for Delta table queries.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::logical_plan::TableScanBuilder;
use datafusion::logical_expr::statistics::StatisticsRequest;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use tracing::debug;

use super::DeltaScanNext;

/// Requests Delta statistics for each query during logical optimization.
///
/// [`DeltaSessionContext`](super::DeltaSessionContext) installs this rule by default. Custom
/// DataFusion sessions can install it on their [`SessionState`](datafusion::execution::SessionState)
/// with `with_optimizer_rule(Arc::new(DeltaStatisticsRule))`.
///
/// The rule requests row count and physical byte estimates for each Delta scan. It requests
/// minimum, maximum, and null count metadata for columns that filters, joins, or ordering
/// expressions reference. The rule follows projection and subquery alias lineage back to the
/// source scan. Delta reads these values from transaction log `Add` metadata. Planning reads no
/// Parquet files. The provider marks missing or invalid metadata as inexact or absent. It marks
/// column statistics from files with deleted rows as inexact.
#[derive(Debug, Default)]
pub struct DeltaStatisticsRule;

impl DeltaStatisticsRule {
    fn collect_columns(exprs: impl IntoIterator<Item = Expr>, columns: &mut HashSet<Column>) {
        for expr in exprs {
            columns.extend(expr.column_refs().iter().map(|column| (*column).clone()));
        }
    }

    fn collect_requested_columns(plan: &LogicalPlan, columns: &mut HashSet<Column>) {
        match plan {
            LogicalPlan::Filter(_) | LogicalPlan::Join(_) | LogicalPlan::Sort(_) => {
                Self::collect_columns(plan.expressions(), columns);
            }
            _ => {}
        }
        for input in plan.inputs() {
            Self::collect_requested_columns(input, columns);
        }
    }

    fn propagate_column_lineage(plan: &LogicalPlan, columns: &mut HashSet<Column>) {
        // Walk from each consumer toward its inputs and carry new source columns through the
        // remaining projections.
        match plan {
            LogicalPlan::Projection(projection) => {
                let sources = columns
                    .iter()
                    .filter_map(|column| projection.schema.index_of_column(column).ok())
                    .flat_map(|index| projection.expr[index].column_refs())
                    .map(|source| (*source).clone())
                    .collect::<Vec<_>>();
                columns.extend(sources);
            }
            LogicalPlan::SubqueryAlias(alias) => {
                let sources = columns
                    .iter()
                    .filter_map(|column| alias.schema.index_of_column(column).ok())
                    .map(|index| Column::from(alias.input.schema().qualified_field(index)))
                    .collect::<Vec<_>>();
                columns.extend(sources);
            }
            _ => {}
        }
        for input in plan.inputs() {
            Self::propagate_column_lineage(input, columns);
        }
    }

    fn requests_for_scan(
        scan: &datafusion::logical_expr::logical_plan::TableScan,
        columns: &HashSet<Column>,
    ) -> Option<std::collections::BTreeSet<StatisticsRequest>> {
        let provider = source_as_provider(&scan.source).ok()?;
        provider.downcast_ref::<DeltaScanNext>()?;

        let mut requests = scan.statistics_requests.clone();
        requests.insert(StatisticsRequest::RowCount);
        requests.insert(StatisticsRequest::TotalByteSize);
        let requested_names = columns
            .iter()
            .filter_map(|column| {
                scan.projected_schema
                    .index_of_column(column)
                    .ok()
                    .map(|index| scan.projected_schema.field(index).name())
            })
            .collect::<HashSet<_>>();
        for field in scan.source.schema().fields() {
            if !requested_names.contains(field.name()) {
                continue;
            }
            let column = Arc::new(Column::from_name(field.name()));
            requests.insert(StatisticsRequest::Min(Arc::clone(&column)));
            requests.insert(StatisticsRequest::Max(Arc::clone(&column)));
            requests.insert(StatisticsRequest::NullCount(column));
        }
        debug!(
            table = %scan.table_name,
            requested_columns = ?requested_names,
            statistics_requests = ?requests,
            "selected Delta scan statistics requests"
        );
        Some(requests)
    }
}

impl OptimizerRule for DeltaStatisticsRule {
    fn name(&self) -> &str {
        "delta_statistics_requests"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut columns = HashSet::new();
        Self::collect_requested_columns(&plan, &mut columns);
        Self::propagate_column_lineage(&plan, &mut columns);

        plan.transform_up(|node| {
            let LogicalPlan::TableScan(scan) = node else {
                return Ok(Transformed::no(node));
            };
            let Some(requests) = Self::requests_for_scan(&scan, &columns) else {
                return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
            };
            if requests == scan.statistics_requests {
                return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
            }
            let scan = TableScanBuilder::from(scan)
                .with_statistics_requests(requests)
                .build()?;
            Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
        })
    }
}
