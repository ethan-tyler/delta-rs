#![allow(clippy::await_holding_lock)]

use delta_bench::assertions::CaseAssertion;
use delta_bench::cli::RunnerMode;
use delta_bench::data::fixtures::{
    generate_fixtures, generate_fixtures_with_profile, FixtureProfile,
};
use delta_bench::storage::StorageConfig;
use delta_bench::suites::{
    apply_dataset_assertion_policy, plan_run_cases, run_planned_cases, run_target, PlannedCase,
};
use std::future::Future;
use std::sync::{Mutex, OnceLock};

#[test]
fn case_filter_requires_at_least_one_matching_case() {
    let err = plan_run_cases("all", RunnerMode::Rust, Some("definitely_not_a_case"))
        .expect_err("no case match must return explicit error");
    assert!(
        err.to_string().contains("case filter"),
        "unexpected error: {err}"
    );
}

#[test]
fn all_runner_plan_includes_python_manifest_cases() {
    let plan = plan_run_cases("all", RunnerMode::All, None).expect("plan should build");
    let ids = plan.iter().map(|case| case.id.as_str()).collect::<Vec<_>>();

    assert!(
        ids.contains(&"pandas_roundtrip_smoke"),
        "interop python case should be included in runner=all plan"
    );
    assert!(
        ids.contains(&"pyarrow_dataset_scan_perf"),
        "interop python case should be included in runner=all plan"
    );
}

#[test]
fn all_runner_plan_is_manifest_ordered() {
    let plan = plan_run_cases("all", RunnerMode::All, None).expect("plan should build");
    let ids = plan.iter().map(|case| case.id.as_str()).collect::<Vec<_>>();
    let rust_idx = ids
        .iter()
        .position(|id| *id == "scan_full_narrow")
        .expect("rust case missing");
    let py_idx = ids
        .iter()
        .position(|id| *id == "pandas_roundtrip_smoke")
        .expect("python case missing");
    assert!(
        rust_idx < py_idx,
        "expected rust manifest sequence to be emitted before python manifest sequence"
    );
}

#[tokio::test]
async fn run_planned_cases_applies_assertions_and_can_fail_case() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let planned = vec![PlannedCase {
        id: "write_append_small".to_string(),
        target: "write".to_string(),
        assertions: vec![CaseAssertion::ExactResultHash(
            "sha256:not-real".to_string(),
        )],
    }];
    let cases = run_planned_cases(temp.path(), &planned, "sf1", 0, 1, &storage)
        .await
        .expect("planned run should execute");
    let only = &cases[0];

    assert!(!only.success, "assertion should convert case into failure");
    assert!(
        only.failure
            .as_ref()
            .map(|f| f.message.contains("result hash mismatch"))
            .unwrap_or(false),
        "unexpected failure payload: {:?}",
        only.failure
    );
}

#[tokio::test]
async fn run_planned_cases_applies_expected_failure_reclassification() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    let planned = vec![PlannedCase {
        id: "write_append_small".to_string(),
        target: "write".to_string(),
        assertions: vec![CaseAssertion::ExpectedErrorContains(
            "fixture load failed".to_string(),
        )],
    }];
    let cases = run_planned_cases(temp.path(), &planned, "sf1", 0, 1, &storage)
        .await
        .expect("planned run should execute");
    let only = &cases[0];

    assert!(only.success, "expected-error assertion should mark success");
    assert_eq!(only.classification, "expected_failure");
}

#[tokio::test]
async fn manifest_hash_assertions_pass_for_write_case() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let planned = plan_run_cases("write", RunnerMode::Rust, Some("write_append_small"))
        .expect("plan should include write case with assertions");
    assert_eq!(planned.len(), 1);
    assert!(
        planned[0].assertions.len() >= 2,
        "expected exact_result_hash and schema_hash assertions from manifest"
    );

    let cases = run_planned_cases(temp.path(), &planned, "sf1", 0, 1, &storage)
        .await
        .expect("planned run should execute");
    let only = &cases[0];
    assert!(
        only.success,
        "manifest-based hash assertions should pass for write case: {:?}",
        only.failure
    );
}

#[tokio::test]
async fn run_target_all_requires_manifest_planning_api() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    let err = run_target(temp.path(), "all", "sf1", 0, 1, &storage)
        .await
        .expect_err("run_target(all) should be rejected");
    assert!(
        err.to_string().contains("plan_run_cases"),
        "unexpected error: {err}"
    );
}

#[test]
fn tpcds_duckdb_dataset_skips_tpcds_hash_assertions() {
    let mut planned = vec![PlannedCase {
        id: "tpcds_q03".to_string(),
        target: "tpcds".to_string(),
        assertions: vec![
            CaseAssertion::ExactResultHash("sha256:expected".to_string()),
            CaseAssertion::SchemaHash("sha256:schema".to_string()),
            CaseAssertion::ExpectedErrorContains("fixture load failed".to_string()),
        ],
    }];

    apply_dataset_assertion_policy(&mut planned, Some("tpcds_duckdb"));

    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].assertions.len(), 2);
    assert!(
        planned[0]
            .assertions
            .iter()
            .all(|assertion| !matches!(assertion, CaseAssertion::ExactResultHash(_))),
        "exact result hash assertions should be dropped for tpcds_duckdb"
    );
    assert!(
        planned[0]
            .assertions
            .iter()
            .any(|assertion| matches!(assertion, CaseAssertion::SchemaHash(_))),
        "schema hash assertions must remain enabled for tpcds_duckdb"
    );
    assert!(planned[0]
        .assertions
        .iter()
        .any(|assertion| matches!(assertion, CaseAssertion::ExpectedErrorContains(_))));
}

#[test]
fn tpcds_duckdb_dataset_policy_does_not_modify_non_tpcds_cases() {
    let mut planned = vec![PlannedCase {
        id: "write_append_small".to_string(),
        target: "write".to_string(),
        assertions: vec![
            CaseAssertion::ExactResultHash("sha256:expected".to_string()),
            CaseAssertion::SchemaHash("sha256:schema".to_string()),
        ],
    }];

    apply_dataset_assertion_policy(&mut planned, Some("tpcds_duckdb"));

    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].assertions.len(), 2);
    assert!(matches!(
        planned[0].assertions[0],
        CaseAssertion::ExactResultHash(_)
    ));
    assert!(matches!(
        planned[0].assertions[1],
        CaseAssertion::SchemaHash(_)
    ));
}

#[tokio::test]
async fn tpcds_duckdb_schema_hash_mismatch_still_fails_after_policy() {
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

    with_env_vars(
        &[(
            "DELTA_BENCH_TPCDS_DUCKDB_SCRIPT",
            script.to_string_lossy().as_ref(),
        )],
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

    let mut planned = vec![PlannedCase {
        id: "tpcds_q03".to_string(),
        target: "tpcds".to_string(),
        assertions: vec![
            CaseAssertion::ExactResultHash("sha256:any".to_string()),
            CaseAssertion::SchemaHash("sha256:not-real".to_string()),
        ],
    }];

    apply_dataset_assertion_policy(&mut planned, Some("tpcds_duckdb"));
    assert!(
        planned[0]
            .assertions
            .iter()
            .all(|assertion| !matches!(assertion, CaseAssertion::ExactResultHash(_))),
        "exact result hash should be removed for tpcds_duckdb"
    );
    assert!(
        planned[0]
            .assertions
            .iter()
            .any(|assertion| matches!(assertion, CaseAssertion::SchemaHash(_))),
        "schema hash should remain for tpcds_duckdb"
    );

    let cases = run_planned_cases(temp.path(), &planned, "sf1", 0, 1, &storage)
        .await
        .expect("planned run should execute");
    assert_eq!(cases.len(), 1);
    let only = &cases[0];
    assert!(!only.success, "schema hash mismatch should fail");
    let failure = only.failure.as_ref().expect("failure payload");
    assert!(
        failure.message.contains("schema hash mismatch"),
        "unexpected failure payload: {}",
        failure.message
    );
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock poisoned")
}

async fn with_env_vars<F, Fut>(entries: &[(&str, &str)], f: F)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let _restore_guard = EnvVarRestoreGuard::set(entries);
    f().await;
}

struct EnvVarRestoreGuard {
    previous: Vec<(String, Option<std::ffi::OsString>)>,
}

impl EnvVarRestoreGuard {
    fn set(entries: &[(&str, &str)]) -> Self {
        let previous = entries
            .iter()
            .map(|(key, _)| ((*key).to_string(), std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in entries {
            std::env::set_var(key, value);
        }
        Self { previous }
    }
}

impl Drop for EnvVarRestoreGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..) {
            if let Some(value) = value {
                std::env::set_var(&key, value);
            } else {
                std::env::remove_var(&key);
            }
        }
    }
}

#[tokio::test]
async fn with_env_vars_restores_values_when_closure_panics() {
    let _env_lock = env_lock();
    let key = "DELTA_BENCH_TEST_ENV_PANIC_RESTORE";
    let original = std::env::var_os(key);

    let join = tokio::spawn(async move {
        with_env_vars(&[(key, "panic-value")], || async {
            panic!("intentional panic for env restore");
        })
        .await;
    })
    .await;

    let err = join.expect_err("task should panic");
    assert!(err.is_panic(), "unexpected join error: {err}");
    assert_eq!(
        std::env::var_os(key),
        original,
        "env var should be restored even when closure panics"
    );
}
