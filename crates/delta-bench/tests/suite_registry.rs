use delta_bench::suites::{list_cases_for_target, list_targets};

#[test]
fn list_targets_includes_optimize_vacuum() {
    assert!(
        list_targets().contains(&"optimize_vacuum"),
        "optimize_vacuum target missing from list_targets"
    );
}

#[test]
fn list_targets_includes_delete_update() {
    assert!(
        list_targets().contains(&"delete_update"),
        "delete_update target missing from list_targets"
    );
}

#[test]
fn list_targets_includes_tpcds() {
    assert!(
        list_targets().contains(&"tpcds"),
        "tpcds target missing from list_targets"
    );
}

#[test]
fn list_targets_includes_interop_py() {
    assert!(
        list_targets().contains(&"interop_py"),
        "interop_py target missing from list_targets"
    );
}

#[test]
fn optimize_vacuum_case_list_is_exact() {
    let cases = list_cases_for_target("optimize_vacuum").expect("known target should work");
    assert_eq!(
        cases,
        vec![
            "optimize_compact_small_files".to_string(),
            "optimize_noop_already_compact".to_string(),
            "optimize_heavy_compaction".to_string(),
            "vacuum_dry_run_lite".to_string(),
            "vacuum_execute_lite".to_string(),
        ]
    );
}

#[test]
fn scan_case_list_is_exact() {
    let cases = list_cases_for_target("scan").expect("known target should work");
    assert_eq!(
        cases,
        vec![
            "scan_full_narrow".to_string(),
            "scan_projection_region".to_string(),
            "scan_filter_flag".to_string(),
            "scan_pruning_hit".to_string(),
            "scan_pruning_miss".to_string(),
        ]
    );
}

#[test]
fn delete_update_case_list_is_exact() {
    let cases = list_cases_for_target("delete_update").expect("known target should work");
    assert_eq!(
        cases,
        vec![
            "delete_1pct_localized".to_string(),
            "delete_5pct_scattered".to_string(),
            "delete_50pct_broad".to_string(),
            "update_literal_1pct_localized".to_string(),
            "update_literal_5pct_scattered".to_string(),
            "update_expr_50pct_broad".to_string(),
            "update_all_rows_expr".to_string(),
        ]
    );
}

#[test]
fn tpcds_case_list_is_exact() {
    let cases = list_cases_for_target("tpcds").expect("known target should work");
    assert_eq!(
        cases,
        vec![
            "tpcds_q03".to_string(),
            "tpcds_q07".to_string(),
            "tpcds_q64".to_string(),
            "tpcds_q72".to_string(),
        ]
    );
}

#[test]
fn interop_py_case_list_is_exact() {
    let cases = list_cases_for_target("interop_py").expect("known target should work");
    assert_eq!(
        cases,
        vec![
            "pandas_roundtrip_smoke".to_string(),
            "polars_roundtrip_smoke".to_string(),
            "pyarrow_dataset_scan_perf".to_string(),
        ]
    );
}

#[test]
fn all_case_list_includes_interop_py_cases() {
    let cases = list_cases_for_target("all").expect("known target should work");
    assert!(
        cases.iter().any(|case| case == "pandas_roundtrip_smoke"),
        "all target should include interop_py cases"
    );
}

#[test]
fn unknown_target_returns_error() {
    let err = list_cases_for_target("totally_unknown_target")
        .expect_err("unknown target should return an explicit error");
    assert!(
        err.to_string().contains("unknown suite target"),
        "unexpected error: {err}"
    );
}

#[test]
fn legacy_scan_target_alias_resolves_to_scan_cases() {
    let canonical = list_cases_for_target("scan").expect("canonical target should work");
    let legacy = list_cases_for_target("read_scan").expect("legacy target alias should work");
    assert_eq!(legacy, canonical);
}

#[test]
fn legacy_delete_update_target_alias_resolves_to_delete_update_cases() {
    let canonical = list_cases_for_target("delete_update").expect("canonical target should work");
    let legacy =
        list_cases_for_target("delete_update_dml").expect("legacy target alias should work");
    assert_eq!(legacy, canonical);
}

#[test]
fn legacy_merge_target_alias_resolves_to_merge_cases() {
    let canonical = list_cases_for_target("merge").expect("canonical target should work");
    let legacy = list_cases_for_target("merge_dml").expect("legacy target alias should work");
    assert_eq!(legacy, canonical);
}
