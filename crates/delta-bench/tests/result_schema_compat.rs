use delta_bench::results::BenchRunResult;

#[test]
fn schema_v2_fields_parse_and_round_trip() {
    let payload = r#"
{
  "schema_version": 2,
  "context": {
    "schema_version": 2,
    "label": "v2-run",
    "git_sha": "abc123",
    "created_at": "2026-02-27T22:48:22.208400Z",
    "host": "test-host",
    "suite": "merge",
    "scale": "sf1",
    "iterations": 1,
    "warmup": 1,
    "dataset_id": "small_files",
    "dataset_fingerprint": "sha256:abc",
    "runner": "rust",
    "backend_profile": "s3_locking_vultr"
  },
  "cases": [
    {
      "case": "merge_upsert_10pct",
      "success": true,
      "classification": "supported",
      "samples": [
        {
          "elapsed_ms": 9.9,
          "rows": 12,
          "bytes": null,
          "metrics": {
            "rows_processed": 12,
            "bytes_processed": null,
            "operations": 1,
            "table_version": 2,
            "files_scanned": 4,
            "files_pruned": 1,
            "bytes_scanned": 4096,
            "scan_time_ms": 4,
            "rewrite_time_ms": 5,
            "peak_rss_mb": 256,
            "cpu_time_ms": 16,
            "bytes_read": 8192,
            "bytes_written": 4096,
            "files_touched": 9,
            "files_skipped": 3,
            "spill_bytes": 0,
            "result_hash": "sha256:def"
          }
        }
      ],
      "failure": null
    }
  ]
}
"#;

    let parsed: BenchRunResult = serde_json::from_str(payload).expect("schema v2 should parse");
    assert_eq!(parsed.schema_version, 2);
    assert_eq!(parsed.context.schema_version, 2);
    assert_eq!(parsed.context.dataset_id.as_deref(), Some("small_files"));
    assert_eq!(parsed.context.runner.as_deref(), Some("rust"));
    assert_eq!(
        parsed.context.backend_profile.as_deref(),
        Some("s3_locking_vultr")
    );

    assert_eq!(parsed.cases[0].classification, "supported");

    let metrics = parsed.cases[0].samples[0]
        .metrics
        .as_ref()
        .expect("metrics should be present");
    assert_eq!(metrics.peak_rss_mb, Some(256));
    assert_eq!(metrics.cpu_time_ms, Some(16));
    assert_eq!(metrics.bytes_read, Some(8192));
    assert_eq!(metrics.bytes_written, Some(4096));
    assert_eq!(metrics.files_touched, Some(9));
    assert_eq!(metrics.files_skipped, Some(3));
    assert_eq!(metrics.spill_bytes, Some(0));
    assert_eq!(metrics.result_hash.as_deref(), Some("sha256:def"));

    let serialized = serde_json::to_string(&parsed).expect("serialize round-trip");
    let reparsed: BenchRunResult =
        serde_json::from_str(&serialized).expect("reparse round-trip payload");
    assert_eq!(reparsed.schema_version, 2);
    assert_eq!(reparsed.context.schema_version, 2);
}

#[test]
fn schema_v1_payload_is_rejected() {
    let payload = r#"
{
  "schema_version": 1,
  "context": {
    "schema_version": 1,
    "label": "legacy-run",
    "git_sha": "abc123",
    "created_at": "2026-02-27T22:48:22.208400Z",
    "host": "test-host",
    "suite": "scan",
    "scale": "sf1",
    "iterations": 1,
    "warmup": 1
  },
  "cases": []
}
"#;

    let err = serde_json::from_str::<BenchRunResult>(payload).expect_err("v1 must fail");
    assert!(
        err.to_string().contains("schema_version"),
        "unexpected error: {err}"
    );
}

#[test]
fn missing_case_classification_is_rejected() {
    let payload = r#"
{
  "schema_version": 2,
  "context": {
    "schema_version": 2,
    "label": "v2-run",
    "git_sha": "abc123",
    "created_at": "2026-02-27T22:48:22.208400Z",
    "host": "test-host",
    "suite": "merge",
    "scale": "sf1",
    "iterations": 1,
    "warmup": 1
  },
  "cases": [
    {
      "case": "merge_upsert_10pct",
      "success": true,
      "samples": [],
      "failure": null
    }
  ]
}
"#;

    let err = serde_json::from_str::<BenchRunResult>(payload)
        .expect_err("missing classification must fail");
    assert!(
        err.to_string().contains("classification"),
        "unexpected error: {err}"
    );
}

#[test]
fn unknown_case_classification_is_rejected() {
    let payload = r#"
{
  "schema_version": 2,
  "context": {
    "schema_version": 2,
    "label": "v2-run",
    "git_sha": "abc123",
    "created_at": "2026-02-27T22:48:22.208400Z",
    "host": "test-host",
    "suite": "merge",
    "scale": "sf1",
    "iterations": 1,
    "warmup": 1
  },
  "cases": [
    {
      "case": "merge_upsert_10pct",
      "success": true,
      "classification": "experimental",
      "samples": [],
      "failure": null
    }
  ]
}
"#;

    let err = serde_json::from_str::<BenchRunResult>(payload)
        .expect_err("unknown classification must fail");
    assert!(
        err.to_string().contains("classification"),
        "unexpected error: {err}"
    );
}

#[test]
fn schema_v2_elapsed_stats_parse_when_present() {
    let payload = r#"
{
  "schema_version": 2,
  "context": {
    "schema_version": 2,
    "label": "v2-run",
    "git_sha": "abc123",
    "created_at": "2026-02-27T22:48:22.208400Z",
    "host": "test-host",
    "suite": "scan",
    "scale": "sf1",
    "iterations": 3,
    "warmup": 1
  },
  "cases": [
    {
      "case": "scan_full_narrow",
      "success": true,
      "classification": "supported",
      "samples": [
        {
          "elapsed_ms": 9.9,
          "rows": 12,
          "bytes": null,
          "metrics": null
        }
      ],
      "elapsed_stats": {
        "min_ms": 9.9,
        "max_ms": 12.1,
        "mean_ms": 10.5,
        "median_ms": 10.2,
        "stddev_ms": 0.9,
        "cv_pct": 8.57
      },
      "failure": null
    }
  ]
}
"#;

    let parsed: BenchRunResult =
        serde_json::from_str(payload).expect("schema v2 with elapsed_stats should parse");
    let elapsed_stats = parsed.cases[0]
        .elapsed_stats
        .as_ref()
        .expect("elapsed_stats should be present");
    assert_eq!(elapsed_stats.min_ms, 9.9);
    assert_eq!(elapsed_stats.max_ms, 12.1);
    assert_eq!(elapsed_stats.mean_ms, 10.5);
    assert_eq!(elapsed_stats.median_ms, 10.2);
    assert_eq!(elapsed_stats.stddev_ms, 0.9);
    assert_eq!(elapsed_stats.cv_pct, Some(8.57));
}
