use delta_bench::data::fixtures::generate_fixtures;
use delta_bench::storage::StorageConfig;
use delta_bench::suites::{merge, metadata, write};

#[tokio::test]
async fn write_samples_include_normalized_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = write::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run write suite");
    let first_sample = &cases[0].samples[0];
    let metrics = first_sample
        .metrics
        .as_ref()
        .expect("metrics should be present");
    assert!(metrics.rows_processed.is_some());
    assert!(metrics.operations.is_some());
    assert!(metrics.table_version.is_some());
}

#[tokio::test]
async fn metadata_samples_report_table_version_metric() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = metadata::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run metadata suite");
    assert!(!cases.is_empty());
    assert!(
        cases[0].success,
        "metadata case failed: {:?}",
        cases[0].failure
    );
    let sample = &cases[0].samples[0];
    let metrics = sample.metrics.as_ref().expect("metrics should be present");
    assert!(metrics.table_version.is_some());
    assert_eq!(sample.rows, metrics.rows_processed);
}

#[tokio::test]
async fn merge_samples_include_merge_scan_and_rewrite_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = merge::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run merge suite");
    assert!(
        cases.iter().any(|case| case.case == "merge_localized_1pct"),
        "expected merge_localized_1pct case; cases={:?}",
        cases.iter().map(|case| &case.case).collect::<Vec<_>>()
    );
    let sample_metrics = cases
        .iter()
        .filter(|case| case.success)
        .flat_map(|case| case.samples.iter())
        .filter_map(|sample| sample.metrics.as_ref())
        .collect::<Vec<_>>();
    assert!(
        !sample_metrics.is_empty(),
        "expected merge sample metrics; cases={:?}",
        cases
            .iter()
            .map(|case| (&case.case, case.success, &case.failure))
            .collect::<Vec<_>>()
    );
    assert!(sample_metrics.iter().any(|m| m.files_scanned.is_some()));
    assert!(sample_metrics.iter().any(|m| m.files_pruned.is_some()));
    assert!(sample_metrics.iter().any(|m| m.scan_time_ms.is_some()));
    assert!(sample_metrics.iter().any(|m| m.rewrite_time_ms.is_some()));
}
