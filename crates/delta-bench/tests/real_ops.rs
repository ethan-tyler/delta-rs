use delta_bench::data::fixtures::generate_fixtures;
use delta_bench::storage::StorageConfig;
use delta_bench::suites::{merge, optimize_vacuum, scan};

#[tokio::test]
async fn generated_fixtures_support_real_scan_suite() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = scan::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("scan suite run");
    assert!(!cases.is_empty());
    assert!(cases.iter().all(|c| c.success));
}

#[tokio::test]
async fn scan_samples_include_physical_scan_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = scan::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("scan suite run");
    assert!(!cases.is_empty());
    let sample_metrics = cases
        .iter()
        .filter(|case| case.success)
        .flat_map(|case| case.samples.iter())
        .filter_map(|sample| sample.metrics.as_ref())
        .collect::<Vec<_>>();
    assert!(
        !sample_metrics.is_empty(),
        "expected scan sample metrics; cases={:?}",
        cases
            .iter()
            .map(|case| (&case.case, case.success, &case.failure))
            .collect::<Vec<_>>()
    );
    assert!(sample_metrics.iter().any(|m| m.files_scanned.is_some()));
    assert!(sample_metrics.iter().any(|m| m.bytes_scanned.is_some()));
    assert!(sample_metrics.iter().any(|m| m.scan_time_ms.is_some()));
    for metrics in sample_metrics {
        if let (Some(files_scanned), Some(files_pruned)) =
            (metrics.files_scanned, metrics.files_pruned)
        {
            assert!(
                files_scanned + files_pruned > 0,
                "expected non-zero scan accounting"
            );
        }
    }
}

#[tokio::test]
async fn scan_pruning_hit_scans_fewer_files_than_miss() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = scan::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("scan suite run");

    let hit_case = cases
        .iter()
        .find(|case| case.case == "scan_pruning_hit")
        .expect("expected scan_pruning_hit case");
    let miss_case = cases
        .iter()
        .find(|case| case.case == "scan_pruning_miss")
        .expect("expected scan_pruning_miss case");

    assert!(
        hit_case.success,
        "hit case should succeed: {:?}",
        hit_case.failure
    );
    assert!(
        miss_case.success,
        "miss case should succeed: {:?}",
        miss_case.failure
    );

    let hit_metrics = hit_case
        .samples
        .first()
        .and_then(|sample| sample.metrics.as_ref())
        .expect("expected metrics for hit case");
    let miss_metrics = miss_case
        .samples
        .first()
        .and_then(|sample| sample.metrics.as_ref())
        .expect("expected metrics for miss case");

    let hit_scanned = hit_metrics
        .files_scanned
        .expect("hit files_scanned should be present");
    let miss_scanned = miss_metrics
        .files_scanned
        .expect("miss files_scanned should be present");
    let hit_pruned = hit_metrics
        .files_pruned
        .expect("hit files_pruned should be present");
    let miss_pruned = miss_metrics
        .files_pruned
        .expect("miss files_pruned should be present");

    assert!(
        hit_scanned < miss_scanned,
        "expected hit files_scanned < miss files_scanned, got {hit_scanned} vs {miss_scanned}"
    );
    assert!(
        hit_pruned > miss_pruned,
        "expected hit files_pruned > miss files_pruned, got {hit_pruned} vs {miss_pruned}"
    );
}

#[tokio::test]
async fn generated_fixtures_support_optimize_vacuum_suite() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = optimize_vacuum::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("optimize_vacuum suite run");
    assert_eq!(cases.len(), 5);
    assert!(
        cases.iter().all(|c| c.success),
        "optimize_vacuum failures: {:?}",
        cases
            .iter()
            .map(|c| (&c.case, &c.failure))
            .collect::<Vec<_>>()
    );

    let optimize_case = cases
        .iter()
        .find(|c| c.case == "optimize_compact_small_files")
        .expect("optimize case should exist");
    let optimize_sample = optimize_case
        .samples
        .first()
        .expect("optimize sample should exist");
    let optimize_metrics = optimize_sample
        .metrics
        .as_ref()
        .expect("optimize metrics should exist");
    let files_scanned = optimize_metrics
        .files_scanned
        .expect("files_scanned should be present");
    let files_pruned = optimize_metrics
        .files_pruned
        .expect("files_pruned should be present");
    assert!(
        files_scanned >= files_pruned,
        "files_scanned should be >= files_pruned"
    );
    let compact_operations = optimize_metrics
        .operations
        .expect("compact optimize operations should be present");

    let noop_case = cases
        .iter()
        .find(|c| c.case == "optimize_noop_already_compact")
        .expect("noop optimize case should exist");
    let noop_metrics = noop_case
        .samples
        .first()
        .and_then(|sample| sample.metrics.as_ref())
        .expect("noop optimize metrics should exist");
    assert_eq!(
        noop_metrics.operations,
        Some(0),
        "noop optimize should not rewrite files"
    );
    assert_eq!(
        noop_metrics.files_scanned, noop_metrics.files_pruned,
        "noop optimize should skip all scanned files"
    );

    let heavy_case = cases
        .iter()
        .find(|c| c.case == "optimize_heavy_compaction")
        .expect("heavy optimize case should exist");
    let heavy_metrics = heavy_case
        .samples
        .first()
        .and_then(|sample| sample.metrics.as_ref())
        .expect("heavy optimize metrics should exist");
    let heavy_operations = heavy_metrics
        .operations
        .expect("heavy optimize operations should be present");
    assert!(heavy_operations > 0, "heavy optimize should rewrite files");
    assert!(
        heavy_operations > compact_operations,
        "heavy optimize should perform more rewrite operations than compact case, got {heavy_operations} vs {compact_operations}"
    );
    assert!(
        heavy_metrics.files_scanned.unwrap_or(0) >= heavy_metrics.files_pruned.unwrap_or(0),
        "heavy optimize should not prune more files than it scanned"
    );
}

#[tokio::test]
async fn merge_partition_localized_case_reports_pruned_files() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = merge::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run merge suite");
    let localized = cases
        .iter()
        .find(|case| case.case == "merge_localized_1pct")
        .expect("expected merge_localized_1pct case");
    assert!(
        localized.success,
        "localized merge case failed: {:?}",
        localized.failure
    );
    let metrics = localized
        .samples
        .first()
        .and_then(|sample| sample.metrics.as_ref())
        .expect("expected localized merge metrics");
    assert!(metrics.files_scanned.is_some());
    assert!(metrics.files_pruned.is_some());
    assert!(metrics.scan_time_ms.is_some());
    assert!(metrics.rewrite_time_ms.is_some());
    assert!(
        metrics.files_pruned.unwrap_or(0) > 0,
        "expected localized merge to prune files, got {:?}",
        metrics.files_pruned
    );
}
