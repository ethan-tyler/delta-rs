use delta_bench::manifests::{
    ensure_required_manifests_exist_under_root, load_manifest, DatasetId, ManifestAssertion,
};
use delta_bench::suites::list_cases_for_target;
use delta_bench::suites::tpcds::catalog::phase1_query_catalog;

#[test]
fn loads_p0_rust_manifest_in_file_order() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join("bench/manifests/core_rust.yaml");
    let manifest = load_manifest(&manifest_path).expect("manifest should load");
    let ids = manifest
        .cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        ids,
        vec![
            "scan_full_narrow",
            "scan_projection_region",
            "scan_filter_flag",
            "scan_pruning_hit",
            "scan_pruning_miss",
            "write_append_small",
            "write_append_large",
            "write_overwrite",
            "delete_1pct_localized",
            "delete_5pct_scattered",
            "delete_50pct_broad",
            "update_literal_1pct_localized",
            "update_literal_5pct_scattered",
            "update_expr_50pct_broad",
            "update_all_rows_expr",
            "merge_delete_5pct",
            "merge_upsert_10pct_insert_10pct",
            "merge_upsert_10pct",
            "merge_upsert_50pct",
            "merge_upsert_90pct",
            "merge_localized_1pct",
            "metadata_load",
            "metadata_time_travel_v0",
            "optimize_compact_small_files",
            "optimize_noop_already_compact",
            "optimize_heavy_compaction",
            "vacuum_dry_run_lite",
            "vacuum_execute_lite",
            "tpcds_q03",
            "tpcds_q07",
            "tpcds_q64",
        ]
    );
}

#[test]
fn dataset_id_parsing_supports_p0_catalog() {
    assert_eq!(
        DatasetId::parse("tiny_smoke").expect("tiny_smoke"),
        DatasetId::TinySmoke
    );
    assert_eq!(
        DatasetId::parse("medium_selective").expect("medium_selective"),
        DatasetId::MediumSelective
    );
    assert_eq!(
        DatasetId::parse("small_files").expect("small_files"),
        DatasetId::SmallFiles
    );
    assert_eq!(
        DatasetId::parse("many_versions").expect("many_versions"),
        DatasetId::ManyVersions
    );
    assert_eq!(
        DatasetId::parse("tpcds_duckdb").expect("tpcds_duckdb"),
        DatasetId::TpcdsDuckdb
    );
    assert_eq!(DatasetId::TinySmoke.fixture_profile(), "standard");
    assert_eq!(DatasetId::ManyVersions.fixture_profile(), "many_versions");
    assert_eq!(DatasetId::TpcdsDuckdb.fixture_profile(), "tpcds_duckdb");
    assert_eq!(DatasetId::TpcdsDuckdb.scale(), "sf1");
}

#[test]
fn dataset_id_rejects_unknown_values() {
    let err = DatasetId::parse("not-real").expect_err("unknown dataset id must fail");
    assert!(
        err.to_string().contains("dataset_id"),
        "unexpected error: {err}"
    );
}

#[test]
fn manifest_assertions_parse_from_yaml() {
    let temp = tempfile::tempdir().expect("tempdir");
    let file = temp.path().join("manifest.yaml");
    std::fs::write(
        &file,
        r#"
id: test
description: test manifest
cases:
  - id: case1
    target: write
    runner: rust
    enabled: true
    assertions:
      - type: expected_error_contains
        value: fixture load failed
      - type: version_monotonicity
"#,
    )
    .expect("write manifest");

    let manifest = load_manifest(&file).expect("manifest should parse");
    assert_eq!(manifest.cases.len(), 1);
    assert_eq!(manifest.cases[0].assertions.len(), 2);
    assert!(matches!(
        manifest.cases[0].assertions[0],
        ManifestAssertion::ExpectedErrorContains { .. }
    ));
    assert!(matches!(
        manifest.cases[0].assertions[1],
        ManifestAssertion::VersionMonotonicity
    ));
}

#[test]
fn p0_rust_manifest_includes_all_delete_update_cases() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join("bench/manifests/core_rust.yaml");
    let manifest = load_manifest(&manifest_path).expect("manifest should load");
    let expected_cases = list_cases_for_target("delete_update")
        .expect("delete_update should be a registered suite target");

    for case in expected_cases {
        let present = manifest
            .cases
            .iter()
            .any(|entry| entry.target == "delete_update" && entry.id == case);
        assert!(
            present,
            "missing delete_update manifest entry for case '{case}'"
        );
    }
}

#[test]
fn p0_rust_manifest_enforces_hash_assertions_for_every_enabled_case() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join("bench/manifests/core_rust.yaml");
    let manifest = load_manifest(&manifest_path).expect("manifest should load");

    let missing = manifest
        .cases
        .iter()
        .filter(|case| case.enabled && case.runner == "rust")
        .filter(|case| {
            let has_result_hash = case
                .assertions
                .iter()
                .any(|assertion| matches!(assertion, ManifestAssertion::ExactResultHash { .. }));
            let has_schema_hash = case
                .assertions
                .iter()
                .any(|assertion| matches!(assertion, ManifestAssertion::SchemaHash { .. }));
            !(has_result_hash && has_schema_hash)
        })
        .map(|case| case.id.clone())
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty(),
        "every enabled rust case in core_rust.yaml should include both exact_result_hash and schema_hash assertions, missing={missing:?}"
    );
}

#[test]
fn p0_rust_manifest_case_ids_match_suite_case_lists() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join("bench/manifests/core_rust.yaml");
    let manifest = load_manifest(&manifest_path).expect("manifest should load");

    for case in manifest
        .cases
        .iter()
        .filter(|entry| entry.enabled && entry.runner == "rust")
    {
        let suite_cases = list_cases_for_target(&case.target).unwrap_or_else(|error| {
            panic!(
                "manifest case '{}' references unknown target '{}': {error}",
                case.id, case.target
            )
        });
        assert!(
            suite_cases.contains(&case.id),
            "manifest case '{}' for target '{}' is missing from suite case_names(); suite_cases={suite_cases:?}",
            case.id,
            case.target
        );
    }
}

#[test]
fn p0_rust_manifest_includes_enabled_tpcds_cases() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join("bench/manifests/core_rust.yaml");
    let manifest = load_manifest(&manifest_path).expect("manifest should load");

    for spec in phase1_query_catalog()
        .into_iter()
        .filter(|spec| spec.enabled)
    {
        let case_id = format!("tpcds_{}", spec.id);
        let present = manifest
            .cases
            .iter()
            .any(|entry| entry.target == "tpcds" && entry.id == case_id);
        assert!(
            present,
            "missing enabled TPC-DS manifest entry for case '{case_id}'"
        );
    }
}

#[test]
fn required_manifest_preflight_reports_missing_files_with_actionable_message() {
    let temp = tempfile::tempdir().expect("tempdir");
    let err = ensure_required_manifests_exist_under_root(temp.path())
        .expect_err("missing manifests should fail preflight");
    let message = err.to_string();
    assert!(
        message.contains("core_rust.yaml"),
        "missing rust manifest should be called out: {message}"
    );
    assert!(
        message.contains("core_python.yaml"),
        "missing python manifest should be called out: {message}"
    );
    assert!(
        message.contains("bench/manifests"),
        "error should explain where files belong: {message}"
    );
}
