use std::fs;

use delta_bench::suites::tpcds::catalog::{phase1_query_catalog, TpcdsQuerySpec};
use delta_bench::suites::tpcds::sql_loader::{load_enabled_queries, load_enabled_queries_from_dir};

#[test]
fn loader_returns_sql_for_enabled_phase1_queries() {
    let loaded = load_enabled_queries(&phase1_query_catalog()).expect("load phase1 sql");
    assert!(!loaded.is_empty(), "expected at least one enabled query");
    assert!(loaded.iter().all(|query| !query.sql.trim().is_empty()));
    assert!(
        loaded.iter().all(|query| query.id != "q72"),
        "disabled query q72 should not be loaded"
    );
}

#[test]
fn missing_sql_file_returns_actionable_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    let specs = vec![TpcdsQuerySpec {
        id: "q99",
        sql_file: "q99.sql",
        enabled: true,
        skip_reason: None,
    }];

    let err = load_enabled_queries_from_dir(&specs, temp.path())
        .expect_err("missing file should produce explicit failure");
    let msg = err.to_string();
    assert!(
        msg.contains("q99")
            && msg.contains("q99.sql")
            && msg.contains(&temp.path().display().to_string()),
        "error should include query id and absolute path; got: {msg}"
    );
}

#[test]
fn disabled_queries_are_not_loaded_or_required() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::write(temp.path().join("q03.sql"), "SELECT 1 AS one").expect("write sql");

    let specs = vec![
        TpcdsQuerySpec {
            id: "q03",
            sql_file: "q03.sql",
            enabled: true,
            skip_reason: None,
        },
        TpcdsQuerySpec {
            id: "q72",
            sql_file: "q72.sql",
            enabled: false,
            skip_reason: Some("known issue"),
        },
    ];

    let loaded = load_enabled_queries_from_dir(&specs, temp.path()).expect("load enabled");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, "q03");
}
