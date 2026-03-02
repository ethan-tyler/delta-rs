use delta_bench::data::fixtures::generate_fixtures;
use delta_bench::results::CaseResult;
use delta_bench::storage::StorageConfig;
use delta_bench::suites::run_target;

async fn run_delete_update_suite_once() -> Vec<CaseResult> {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let fixtures_dir = temp.path().join("fixtures");
    let storage = StorageConfig::local();

    generate_fixtures(&fixtures_dir, "sf1", 42, true, &storage)
        .await
        .expect("fixtures should be generated");

    run_target(&fixtures_dir, "delete_update", "sf1", 0, 1, &storage)
        .await
        .expect("delete_update suite should run")
}

#[tokio::test]
async fn delete_update_delete_family_smoke() {
    let cases = run_delete_update_suite_once().await;
    let delete_cases: Vec<&CaseResult> = cases
        .iter()
        .filter(|case| case.case.starts_with("delete_"))
        .collect();
    assert!(
        !delete_cases.is_empty(),
        "expected at least one delete case in delete_update suite"
    );
    for case in delete_cases {
        assert!(
            case.success,
            "delete case should succeed: {} ({:?})",
            case.case, case.failure
        );
    }
}

#[tokio::test]
async fn delete_update_update_family_smoke() {
    let cases = run_delete_update_suite_once().await;
    let update_cases: Vec<&CaseResult> = cases
        .iter()
        .filter(|case| case.case.starts_with("update_"))
        .collect();
    assert!(
        !update_cases.is_empty(),
        "expected at least one update case in delete_update suite"
    );
    for case in update_cases {
        assert!(
            case.success,
            "update case should succeed: {} ({:?})",
            case.case, case.failure
        );
    }
}

#[tokio::test]
async fn delete_update_does_not_depend_on_merge_partitioned_fixture() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let fixtures_dir = temp.path().join("fixtures");
    let storage = StorageConfig::local();

    generate_fixtures(&fixtures_dir, "sf1", 42, true, &storage)
        .await
        .expect("fixtures should be generated");

    let merge_partitioned = fixtures_dir
        .join("sf1")
        .join("merge_partitioned_target_delta");
    std::fs::remove_dir_all(&merge_partitioned).expect("remove merge fixture dir");

    let cases = run_target(&fixtures_dir, "delete_update", "sf1", 0, 1, &storage)
        .await
        .expect("delete_update suite should run");
    assert!(
        !cases.is_empty(),
        "expected delete_update suite to return cases"
    );
    for case in cases {
        assert!(
            case.success,
            "case should succeed without merge fixture dependency: {} ({:?})",
            case.case, case.failure
        );
    }
}
