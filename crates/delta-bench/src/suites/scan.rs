use std::path::Path;
use std::sync::Arc;

use deltalake_core::datafusion::execution::context::TaskContext;
use deltalake_core::datafusion::physical_plan::collect;
use deltalake_core::datafusion::physical_plan::ExecutionPlan;
use deltalake_core::datafusion::prelude::SessionContext;

use crate::data::fixtures::{narrow_sales_table_url, read_partitioned_table_url};
use crate::error::BenchResult;
use crate::fingerprint::{hash_arrow_schema, hash_record_batches_unordered};
use crate::results::{CaseResult, RuntimeIOMetrics, SampleMetrics, ScanRewriteMetrics};
use crate::runner::{run_case_async_with_async_setup_custom_timing, CaseExecutionResult};
use crate::storage::StorageConfig;
use crate::suites::scan_metrics::extract_scan_metrics;
use url::Url;

pub fn case_names() -> Vec<String> {
    vec![
        "scan_full_narrow".to_string(),
        "scan_projection_region".to_string(),
        "scan_filter_flag".to_string(),
        "scan_pruning_hit".to_string(),
        "scan_pruning_miss".to_string(),
    ]
}

struct PreparedSqlQuery {
    plan: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
    total_active_files: Option<u64>,
}

pub async fn run(
    fixtures_dir: &Path,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    let table_url = narrow_sales_table_url(fixtures_dir, scale, storage)?;
    let partitioned_table_url = read_partitioned_table_url(fixtures_dir, scale, storage)?;

    let mut results = Vec::new();

    let full_scan = run_prepared_case(
        "scan_full_narrow",
        warmup,
        iterations,
        storage,
        table_url.clone(),
        "SELECT COUNT(*) FROM bench",
    )
    .await;
    results.push(into_case_result(full_scan));

    let projection = run_prepared_case(
        "scan_projection_region",
        warmup,
        iterations,
        storage,
        table_url.clone(),
        "SELECT region, SUM(value_i64) FROM bench GROUP BY region",
    )
    .await;
    results.push(into_case_result(projection));

    let filtered = run_prepared_case(
        "scan_filter_flag",
        warmup,
        iterations,
        storage,
        table_url.clone(),
        "SELECT COUNT(*) FROM bench WHERE flag = true AND value_i64 > 0",
    )
    .await;
    results.push(into_case_result(filtered));

    let partition_hit = run_prepared_case(
        "scan_pruning_hit",
        warmup,
        iterations,
        storage,
        partitioned_table_url.clone(),
        "SELECT COUNT(*) FROM bench WHERE region = 'us'",
    )
    .await;
    results.push(into_case_result(partition_hit));

    let partition_miss = run_prepared_case(
        "scan_pruning_miss",
        warmup,
        iterations,
        storage,
        partitioned_table_url,
        "SELECT COUNT(*) FROM bench",
    )
    .await;
    results.push(into_case_result(partition_miss));

    Ok(results)
}

async fn run_prepared_case(
    case_name: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
    table_url: Url,
    sql: &'static str,
) -> CaseExecutionResult {
    run_case_async_with_async_setup_custom_timing(
        case_name,
        warmup,
        iterations,
        || {
            let storage = storage.clone();
            let table_url = table_url.clone();
            async move {
                prepare_sql_query(&storage, table_url, sql)
                    .await
                    .map_err(|e| e.to_string())
            }
        },
        |prepared| async move {
            execute_prepared_query(prepared)
                .await
                .map_err(|e| e.to_string())
        },
    )
    .await
}

async fn prepare_sql_query(
    storage: &StorageConfig,
    table_url: Url,
    sql: &str,
) -> BenchResult<PreparedSqlQuery> {
    let table = storage.open_table(table_url).await?;
    let total_active_files = table
        .snapshot()
        .ok()
        .map(|snapshot| snapshot.log_data().num_files() as u64);
    let ctx = SessionContext::new();
    ctx.register_table("bench", table.table_provider().await?)?;
    let df = ctx.sql(sql).await?;
    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await?;

    Ok(PreparedSqlQuery {
        plan,
        task_ctx,
        total_active_files,
    })
}

async fn execute_prepared_query(
    prepared: PreparedSqlQuery,
) -> BenchResult<(SampleMetrics, Option<f64>)> {
    let query_start = std::time::Instant::now();
    let batches = collect(prepared.plan.clone(), prepared.task_ctx).await?;
    let query_elapsed_ms = query_start.elapsed().as_secs_f64() * 1000.0;
    let rows_processed = batches.iter().map(|b| b.num_rows() as u64).sum::<u64>();
    let scan_metrics = extract_scan_metrics(&prepared.plan);
    let files_pruned = scan_metrics.files_pruned.or_else(|| {
        prepared.total_active_files.and_then(|total| {
            scan_metrics
                .files_scanned
                .and_then(|scanned| total.checked_sub(scanned))
        })
    });
    let result_hash = hash_record_batches_unordered(&batches)?;
    let schema_hash = hash_arrow_schema(prepared.plan.schema().as_ref())?;

    Ok((
        SampleMetrics::base(Some(rows_processed), None, None, None)
            .with_scan_rewrite(ScanRewriteMetrics {
                files_scanned: scan_metrics.files_scanned,
                files_pruned,
                bytes_scanned: scan_metrics.bytes_scanned,
                scan_time_ms: scan_metrics.scan_time_ms,
                rewrite_time_ms: None,
            })
            .with_runtime_io(RuntimeIOMetrics {
                peak_rss_mb: None,
                cpu_time_ms: None,
                bytes_read: None,
                bytes_written: None,
                files_touched: None,
                files_skipped: None,
                spill_bytes: None,
                result_hash: Some(result_hash),
                schema_hash: Some(schema_hash),
            }),
        Some(query_elapsed_ms),
    ))
}

fn into_case_result(result: CaseExecutionResult) -> CaseResult {
    match result {
        CaseExecutionResult::Success(c) | CaseExecutionResult::Failure(c) => c,
    }
}

#[cfg(test)]
mod tests {
    use super::{execute_prepared_query, prepare_sql_query};
    use crate::data::fixtures::generate_fixtures;
    use crate::storage::StorageConfig;

    #[tokio::test]
    async fn prepared_query_setup_and_execute_path_produces_metrics() {
        let temp = tempfile::tempdir().expect("tempdir");
        let storage = StorageConfig::local();
        generate_fixtures(temp.path(), "sf1", 42, true, &storage)
            .await
            .expect("generate fixtures");
        let table_url = crate::data::fixtures::narrow_sales_table_url(temp.path(), "sf1", &storage)
            .expect("table URL");

        let prepared = prepare_sql_query(&storage, table_url, "SELECT COUNT(*) FROM bench")
            .await
            .expect("prepare query");
        let (metrics, elapsed_override) = execute_prepared_query(prepared)
            .await
            .expect("execute query");

        assert!(elapsed_override.is_some());
        assert!(metrics.rows_processed.is_some());
        assert!(metrics.rows_processed.unwrap_or(0) > 0);
    }
}
