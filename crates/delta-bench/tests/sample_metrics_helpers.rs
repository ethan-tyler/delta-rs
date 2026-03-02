use delta_bench::results::{RuntimeIOMetrics, SampleMetrics, ScanRewriteMetrics};

#[test]
fn base_constructor_defaults_scan_and_rewrite_metrics_to_none() {
    let metrics = SampleMetrics::base(Some(11), None, Some(2), Some(5));

    assert_eq!(metrics.rows_processed, Some(11));
    assert_eq!(metrics.bytes_processed, None);
    assert_eq!(metrics.operations, Some(2));
    assert_eq!(metrics.table_version, Some(5));
    assert_eq!(metrics.files_scanned, None);
    assert_eq!(metrics.files_pruned, None);
    assert_eq!(metrics.bytes_scanned, None);
    assert_eq!(metrics.scan_time_ms, None);
    assert_eq!(metrics.rewrite_time_ms, None);
    assert_eq!(metrics.result_hash, None);
    assert_eq!(metrics.schema_hash, None);
}

#[test]
fn scan_rewrite_builder_populates_optional_fields() {
    let metrics = SampleMetrics::base(Some(20), None, Some(1), Some(7)).with_scan_rewrite(
        ScanRewriteMetrics {
            files_scanned: Some(8),
            files_pruned: Some(3),
            bytes_scanned: Some(4096),
            scan_time_ms: Some(14),
            rewrite_time_ms: Some(22),
        },
    );

    assert_eq!(metrics.rows_processed, Some(20));
    assert_eq!(metrics.operations, Some(1));
    assert_eq!(metrics.table_version, Some(7));
    assert_eq!(metrics.files_scanned, Some(8));
    assert_eq!(metrics.files_pruned, Some(3));
    assert_eq!(metrics.bytes_scanned, Some(4096));
    assert_eq!(metrics.scan_time_ms, Some(14));
    assert_eq!(metrics.rewrite_time_ms, Some(22));
}

#[test]
fn runtime_io_builder_populates_result_and_schema_hashes() {
    let metrics = SampleMetrics::base(Some(3), Some(99), Some(1), Some(2)).with_runtime_io(
        RuntimeIOMetrics {
            peak_rss_mb: Some(12),
            cpu_time_ms: Some(23),
            bytes_read: Some(34),
            bytes_written: Some(45),
            files_touched: Some(56),
            files_skipped: Some(67),
            spill_bytes: Some(78),
            result_hash: Some("sha256:result".to_string()),
            schema_hash: Some("sha256:schema".to_string()),
        },
    );

    assert_eq!(metrics.result_hash.as_deref(), Some("sha256:result"));
    assert_eq!(metrics.schema_hash.as_deref(), Some("sha256:schema"));
}
