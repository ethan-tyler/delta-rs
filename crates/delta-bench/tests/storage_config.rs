use std::collections::HashMap;
use std::path::Path;

use delta_bench::cli::StorageBackend;
use delta_bench::storage::StorageConfig;

#[test]
fn non_local_storage_requires_table_root_option() {
    let err = StorageConfig::new(StorageBackend::S3, HashMap::new())
        .expect_err("non-local backend without table_root must fail");
    assert!(
        err.to_string().contains("table_root"),
        "unexpected error: {err}"
    );
}

#[test]
fn local_storage_does_not_require_table_root_option() {
    StorageConfig::new(StorageBackend::Local, HashMap::new())
        .expect("local backend should be backward compatible");
}

#[test]
fn non_local_storage_builds_fixture_table_urls() {
    let mut options = HashMap::new();
    options.insert(
        "table_root".to_string(),
        "s3://bench-bucket/delta-bench".to_string(),
    );
    options.insert("AWS_REGION".to_string(), "us-east-1".to_string());

    let config = StorageConfig::new(StorageBackend::S3, options).expect("valid storage config");
    let url = config
        .fixture_table_url("sf1", "narrow_sales_delta")
        .expect("fixture url");
    assert_eq!(
        url.as_str(),
        "s3://bench-bucket/delta-bench/sf1/narrow_sales_delta"
    );
}

#[test]
fn s3_backend_rejects_non_s3_table_root_scheme() {
    let mut options = HashMap::new();
    options.insert("table_root".to_string(), "gs://bucket/path".to_string());

    let err = StorageConfig::new(StorageBackend::S3, options)
        .expect_err("s3 backend should reject non-s3 table_root scheme");
    assert!(
        err.to_string().contains("table_root") && err.to_string().contains("s3"),
        "unexpected error: {err}"
    );
}

#[test]
fn non_local_storage_can_produce_unique_isolated_table_urls() {
    let mut options = HashMap::new();
    options.insert(
        "table_root".to_string(),
        "s3://bench-bucket/delta-bench".to_string(),
    );
    let config = StorageConfig::new(StorageBackend::S3, options).expect("valid storage config");

    let first = config
        .isolated_table_url("sf1", "merge_target_delta", "case/with spaces")
        .expect("first isolated url");
    let second = config
        .isolated_table_url("sf1", "merge_target_delta", "case/with spaces")
        .expect("second isolated url");

    assert_ne!(first, second, "isolated URLs must be unique");
    assert!(first
        .as_str()
        .starts_with("s3://bench-bucket/delta-bench/sf1/"));
    assert!(!first.as_str().contains(" "));
    assert!(!first.as_str().contains("/case/with spaces"));
}

#[test]
fn local_storage_resolves_relative_paths_for_file_urls() {
    let config = StorageConfig::local();
    let relative = Path::new("fixtures/sf1/narrow_sales_delta");

    let url = config
        .table_url_for(relative, "sf1", "narrow_sales_delta")
        .expect("relative local paths should convert to file URLs");
    assert_eq!(url.scheme(), "file");

    let url_path = url.to_file_path().expect("file URL should convert to path");
    let cwd = std::env::current_dir().expect("cwd");
    assert!(
        url_path.starts_with(&cwd),
        "expected URL path '{:?}' to be under cwd '{:?}'",
        url_path,
        cwd
    );
    assert!(
        url_path.ends_with(relative),
        "expected URL path '{:?}' to end with '{:?}'",
        url_path,
        relative
    );
}
