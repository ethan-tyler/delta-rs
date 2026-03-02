#![cfg(unix)]

use std::os::unix::fs::symlink;

use delta_bench::data::fixtures::{generate_fixtures, narrow_sales_table_path};
use delta_bench::storage::StorageConfig;
use delta_bench::suites::metadata;

#[tokio::test]
async fn metadata_suite_rejects_symlink_entries_in_fixture_tree() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let table_path = narrow_sales_table_path(temp.path(), "sf1").expect("table path");
    let link_path = table_path.join("symlink_escape");
    symlink(temp.path(), &link_path).expect("create symlink");

    let cases = metadata::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("metadata suite should produce per-case failures");

    assert!(!cases.is_empty());
    assert!(cases.iter().all(|c| !c.success));
    assert!(cases.iter().all(|c| {
        c.failure
            .as_ref()
            .map(|f| f.message.contains("symlinks are not allowed"))
            .unwrap_or(false)
    }));
}
