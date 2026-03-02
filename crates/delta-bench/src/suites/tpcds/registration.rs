use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use deltalake_core::datafusion::prelude::SessionContext;
use deltalake_core::datafusion::sql::sqlparser::ast::{
    ObjectName, Query, SetExpr, Statement, TableFactor, TableWithJoins,
};
use deltalake_core::datafusion::sql::sqlparser::dialect::GenericDialect;
use deltalake_core::datafusion::sql::sqlparser::parser::Parser;

use crate::error::{BenchError, BenchResult};
use crate::storage::StorageConfig;

const TPCDS_DIR: &str = "tpcds";

pub async fn register_tables_for_sql(
    ctx: &SessionContext,
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
    sql: &str,
) -> BenchResult<()> {
    let table_names = referenced_table_names(sql)?;
    if table_names.is_empty() {
        return Err(BenchError::InvalidArgument(
            "no table references found in TPC-DS SQL".to_string(),
        ));
    }

    for table_name in table_names {
        register_table(ctx, fixtures_dir, scale, storage, &table_name).await?;
    }
    Ok(())
}

async fn register_table(
    ctx: &SessionContext,
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
    table_name: &str,
) -> BenchResult<()> {
    let local_table_path = fixtures_dir.join(scale).join(TPCDS_DIR).join(table_name);
    let remote_table_name = format!("{TPCDS_DIR}/{table_name}");
    let table_url = storage.table_url_for(&local_table_path, scale, &remote_table_name)?;
    let table = storage.open_table(table_url).await?;
    let provider = table.table_provider().await?;
    ctx.register_table(table_name, provider)?;
    Ok(())
}

fn referenced_table_names(sql: &str) -> BenchResult<Vec<String>> {
    let mut names = BTreeSet::new();
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql).map_err(|err| {
        BenchError::InvalidArgument(format!(
            "failed to parse TPC-DS SQL for table registration: {err}"
        ))
    })?;
    let mut cte_scopes = Vec::<HashSet<String>>::new();

    for statement in &statements {
        collect_statement_tables(statement, &mut names, &mut cte_scopes);
    }

    Ok(names.into_iter().collect())
}

fn collect_statement_tables(
    statement: &Statement,
    names: &mut BTreeSet<String>,
    cte_scopes: &mut Vec<HashSet<String>>,
) {
    if let Statement::Query(query) = statement {
        collect_query_tables(query, names, cte_scopes);
    }
}

fn collect_query_tables(
    query: &Query,
    names: &mut BTreeSet<String>,
    cte_scopes: &mut Vec<HashSet<String>>,
) {
    let local_ctes = query
        .with
        .as_ref()
        .map(|with| {
            with.cte_tables
                .iter()
                .map(|cte| cte.alias.name.value.to_ascii_lowercase())
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();
    cte_scopes.push(local_ctes);

    if let Some(with) = &query.with {
        for cte in &with.cte_tables {
            collect_query_tables(&cte.query, names, cte_scopes);
        }
    }

    collect_set_expr_tables(&query.body, names, cte_scopes);
    cte_scopes.pop();
}

fn collect_set_expr_tables(
    set_expr: &SetExpr,
    names: &mut BTreeSet<String>,
    cte_scopes: &mut Vec<HashSet<String>>,
) {
    match set_expr {
        SetExpr::Select(select) => {
            for table_with_joins in &select.from {
                collect_table_with_joins(table_with_joins, names, cte_scopes);
            }
        }
        SetExpr::Query(query) => collect_query_tables(query, names, cte_scopes),
        SetExpr::SetOperation { left, right, .. } => {
            collect_set_expr_tables(left, names, cte_scopes);
            collect_set_expr_tables(right, names, cte_scopes);
        }
        SetExpr::Insert(statement)
        | SetExpr::Update(statement)
        | SetExpr::Delete(statement)
        | SetExpr::Merge(statement) => collect_statement_tables(statement, names, cte_scopes),
        _ => {}
    }
}

fn collect_table_with_joins(
    table_with_joins: &TableWithJoins,
    names: &mut BTreeSet<String>,
    cte_scopes: &mut Vec<HashSet<String>>,
) {
    collect_table_factor(&table_with_joins.relation, names, cte_scopes);
    for join in &table_with_joins.joins {
        collect_table_factor(&join.relation, names, cte_scopes);
    }
}

fn collect_table_factor(
    table_factor: &TableFactor,
    names: &mut BTreeSet<String>,
    cte_scopes: &mut Vec<HashSet<String>>,
) {
    match table_factor {
        TableFactor::Table { name, .. } => {
            if let Some(table_name) = table_name(name) {
                if !is_cte_alias(&table_name, cte_scopes) {
                    names.insert(table_name);
                }
            }
        }
        TableFactor::Derived { subquery, .. } => collect_query_tables(subquery, names, cte_scopes),
        TableFactor::NestedJoin {
            table_with_joins, ..
        } => collect_table_with_joins(table_with_joins, names, cte_scopes),
        TableFactor::Pivot { table, .. }
        | TableFactor::Unpivot { table, .. }
        | TableFactor::MatchRecognize { table, .. } => {
            collect_table_factor(table, names, cte_scopes)
        }
        _ => {}
    }
}

fn table_name(name: &ObjectName) -> Option<String> {
    name.0.iter().rev().find_map(|part| {
        part.as_ident()
            .map(|ident| ident.value.to_ascii_lowercase())
    })
}

fn is_cte_alias(name: &str, cte_scopes: &[HashSet<String>]) -> bool {
    cte_scopes.iter().any(|scope| scope.contains(name))
}

#[cfg(test)]
mod tests {
    use super::referenced_table_names;

    #[test]
    fn extracts_unique_sorted_tables_from_from_and_join_clauses() {
        let sql = r#"
            SELECT ss_item_sk
            FROM store_sales ss
            JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
            JOIN store_sales s2 ON s2.ss_item_sk = ss.ss_item_sk
        "#;
        assert_eq!(
            referenced_table_names(sql).expect("parse sql"),
            vec!["date_dim".to_string(), "store_sales".to_string()]
        );
    }

    #[test]
    fn ignores_cte_aliases_and_keeps_base_tables() {
        let sql = r#"
            WITH sales_subset AS (
                SELECT ss_item_sk, ss_sold_date_sk
                FROM store_sales
                WHERE ss_quantity > 0
            )
            SELECT d.d_date_sk, COUNT(*)
            FROM sales_subset s
            JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
            GROUP BY d.d_date_sk
        "#;
        assert_eq!(
            referenced_table_names(sql).expect("parse sql"),
            vec!["date_dim".to_string(), "store_sales".to_string()]
        );
    }

    #[test]
    fn ignores_derived_table_aliases_and_finds_nested_sources() {
        let sql = r#"
            SELECT COUNT(*)
            FROM (
                SELECT ss_item_sk
                FROM store_sales
                WHERE ss_quantity > 1
            ) derived_sales
            JOIN item i ON derived_sales.ss_item_sk = i.i_item_sk
        "#;
        assert_eq!(
            referenced_table_names(sql).expect("parse sql"),
            vec!["item".to_string(), "store_sales".to_string()]
        );
    }

    #[test]
    fn extracts_tables_from_comma_join_syntax() {
        let sql = r#"
            SELECT COUNT(*)
            FROM store_sales ss, date_dim d
            WHERE ss.ss_sold_date_sk = d.d_date_sk
        "#;
        assert_eq!(
            referenced_table_names(sql).expect("parse sql"),
            vec!["date_dim".to_string(), "store_sales".to_string()]
        );
    }

    #[test]
    fn ignores_table_like_tokens_inside_comments() {
        let sql = r#"
            -- FROM fake_table should be ignored
            SELECT ss_item_sk
            FROM store_sales
            /* JOIN another_fake_table x ON x.id = 1 */
            JOIN date_dim d ON d.d_date_sk = store_sales.ss_sold_date_sk
        "#;
        assert_eq!(
            referenced_table_names(sql).expect("parse sql"),
            vec!["date_dim".to_string(), "store_sales".to_string()]
        );
    }
}
