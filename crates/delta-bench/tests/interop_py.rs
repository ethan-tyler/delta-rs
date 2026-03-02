use delta_bench::data::fixtures::generate_fixtures;
use delta_bench::storage::StorageConfig;
use delta_bench::suites::interop_py;

#[tokio::test]
async fn interop_py_suite_runs_with_deterministic_case_ids() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = interop_py::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("interop suite run");

    assert_eq!(
        cases.iter().map(|c| c.case.as_str()).collect::<Vec<_>>(),
        vec![
            "pandas_roundtrip_smoke",
            "polars_roundtrip_smoke",
            "pyarrow_dataset_scan_perf",
        ]
    );
    assert!(cases.iter().all(|c| c.success));
    assert!(
        cases
            .iter()
            .all(|c| matches!(c.classification.as_str(), "supported" | "expected_failure")),
        "unexpected classifications: {:?}",
        cases
            .iter()
            .map(|c| (&c.case, &c.classification, &c.failure))
            .collect::<Vec<_>>()
    );
}
