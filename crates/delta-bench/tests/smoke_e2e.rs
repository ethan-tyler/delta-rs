use delta_bench::runner::{run_case, CaseExecutionResult};

#[test]
fn runner_records_failures_and_continues() {
    let ok = run_case("ok", 0, 2, || Ok::<u64, String>(10));
    let fail = run_case("fail", 0, 2, || Err::<u64, String>("boom".to_string()));

    match ok {
        CaseExecutionResult::Success(case) => {
            assert_eq!(case.classification, "supported");
        }
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    }

    match fail {
        CaseExecutionResult::Failure(case) => {
            assert_eq!(case.classification, "supported");
        }
        CaseExecutionResult::Success(_) => panic!("expected failure result"),
    }
}
