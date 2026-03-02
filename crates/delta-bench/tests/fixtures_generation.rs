#![allow(clippy::await_holding_lock)]

use delta_bench::data::fixtures::{
    generate_fixtures, generate_fixtures_with_profile, load_manifest, narrow_sales_table_url,
    FixtureProfile,
};
use delta_bench::storage::StorageConfig;
use std::sync::{Mutex, OnceLock};

#[tokio::test]
async fn regenerates_when_requested_seed_changes_without_force() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures(temp.path(), "sf1", 42, false, &storage)
        .await
        .expect("generate fixtures seed 42");
    let first = load_manifest(temp.path(), "sf1").expect("load first manifest");
    assert_eq!(first.seed, 42);

    generate_fixtures(temp.path(), "sf1", 43, false, &storage)
        .await
        .expect("regenerate fixtures seed 43");
    let second = load_manifest(temp.path(), "sf1").expect("load second manifest");
    assert_eq!(second.seed, 43);
}

#[tokio::test]
async fn rejects_unknown_scale() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    let err = generate_fixtures(temp.path(), "sf-nope", 42, false, &storage)
        .await
        .expect_err("unknown scale should fail");
    assert!(
        err.to_string().contains("unknown scale"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn generates_wave1_specialized_fixture_tables() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let root = temp.path().join("sf1");
    for table_name in [
        "read_partitioned_delta",
        "delete_update_small_files_delta",
        "merge_partitioned_target_delta",
        "optimize_compacted_delta",
    ] {
        let table_path = root.join(table_name);
        assert!(
            table_path.exists(),
            "expected table dir: {}",
            table_path.display()
        );
        assert!(
            table_path.join("_delta_log").exists(),
            "expected delta log dir: {}",
            table_path.join("_delta_log").display()
        );
    }
}

#[tokio::test]
async fn regenerates_when_wave1_fixture_tables_are_missing_without_force() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("initial generate fixtures");

    let root = temp.path().join("sf1");
    for table_name in [
        "read_partitioned_delta",
        "merge_partitioned_target_delta",
        "optimize_compacted_delta",
    ] {
        std::fs::remove_dir_all(root.join(table_name))
            .unwrap_or_else(|err| panic!("remove {table_name}: {err}"));
    }

    generate_fixtures(temp.path(), "sf1", 42, false, &storage)
        .await
        .expect("should regenerate when wave1 fixture tables are missing");

    for table_name in [
        "read_partitioned_delta",
        "merge_partitioned_target_delta",
        "optimize_compacted_delta",
    ] {
        let table_path = root.join(table_name);
        assert!(
            table_path.join("_delta_log").exists(),
            "expected regenerated table dir: {}",
            table_path.display()
        );
    }
}

#[tokio::test]
async fn generates_tpcds_store_sales_fixture_table() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let table_path = temp.path().join("sf1").join("tpcds").join("store_sales");
    assert!(
        table_path.exists(),
        "expected TPC-DS store_sales table dir: {}",
        table_path.display()
    );
    assert!(
        table_path.join("_delta_log").exists(),
        "expected TPC-DS store_sales delta log dir: {}",
        table_path.join("_delta_log").display()
    );
}

#[tokio::test]
async fn many_versions_profile_writes_multiple_narrow_sales_table_versions() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures_with_profile(
        temp.path(),
        "sf1",
        42,
        true,
        FixtureProfile::ManyVersions,
        &storage,
    )
    .await
    .expect("generate many-versions fixtures");

    let table_url =
        narrow_sales_table_url(temp.path(), "sf1", &storage).expect("narrow_sales table url");
    let table = storage
        .open_table(table_url)
        .await
        .expect("open many-versions table");
    let version = table.version().map(|v| v as u64).unwrap_or(0);
    assert!(
        version == 12,
        "expected many-versions profile to append 12 commits after initial write, got version={version}"
    );
}

#[tokio::test]
async fn tpcds_duckdb_profile_generates_store_sales_table_via_script_override() {
    let _env_lock = env_lock();
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    let script = temp.path().join("fake_tpcds_generator.py");
    std::fs::write(
        &script,
        r#"#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scale-factor", required=True)
parser.add_argument("--output-csv", required=True)
args = parser.parse_args()

path = Path(args.output_csv)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    "ss_customer_sk,ss_ext_sales_price,ss_item_sk,ss_quantity,ss_sold_date_sk\n"
    "1,11.0,101,2,2450815\n"
    "2,15.5,102,3,2450816\n",
    encoding="utf-8",
)
"#,
    )
    .expect("write fake script");

    with_env_var(
        "DELTA_BENCH_TPCDS_DUCKDB_SCRIPT",
        script.to_string_lossy().as_ref(),
        || async {
            generate_fixtures_with_profile(
                temp.path(),
                "sf1",
                42,
                true,
                FixtureProfile::TpcdsDuckdb,
                &storage,
            )
            .await
            .expect("generate tpcds_duckdb fixtures");
        },
    )
    .await;

    let table_path = temp.path().join("sf1").join("tpcds").join("store_sales");
    assert!(
        table_path.join("_delta_log").exists(),
        "expected DuckDB-backed store_sales delta log: {}",
        table_path.join("_delta_log").display()
    );
    let manifest = load_manifest(temp.path(), "sf1").expect("load fixture manifest");
    assert_eq!(manifest.profile, "tpcds_duckdb");
}

#[tokio::test]
async fn fixture_generation_times_out_when_scale_lock_is_held() {
    let _env_lock = env_lock();
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    let lock_dir = temp.path().join(".delta_bench_locks").join("sf1.lock");
    std::fs::create_dir_all(&lock_dir).expect("create held lock dir");

    with_env_var("DELTA_BENCH_FIXTURE_LOCK_TIMEOUT_MS", "5", || async {
        with_env_var("DELTA_BENCH_FIXTURE_LOCK_RETRY_MS", "1", || async {
            let error = generate_fixtures(temp.path(), "sf1", 42, true, &storage)
                .await
                .expect_err("held lock should force timeout error");
            assert!(
                error
                    .to_string()
                    .contains("timed out waiting for fixture generation lock"),
                "unexpected error: {error}"
            );
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn fixture_generation_creates_and_releases_scale_lock() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();

    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let locks_root = temp.path().join(".delta_bench_locks");
    let scale_lock = locks_root.join("sf1.lock");
    assert!(
        locks_root.exists(),
        "lock root should exist after fixture generation"
    );
    assert!(
        !scale_lock.exists(),
        "scale lock should be released after fixture generation"
    );
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock poisoned")
}

async fn with_env_var<F, Fut>(key: &str, value: &str, f: F)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let previous = std::env::var_os(key);
    std::env::set_var(key, value);
    f().await;
    if let Some(value) = previous {
        std::env::set_var(key, value);
    } else {
        std::env::remove_var(key);
    }
}
