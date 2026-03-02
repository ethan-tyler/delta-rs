use crate::results::{CaseFailure, CaseResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaseAssertion {
    ExactResultHash(String),
    SchemaHash(String),
    ExpectedErrorContains(String),
    VersionMonotonicity,
}

pub fn apply_case_assertions(case: &mut CaseResult, assertions: &[CaseAssertion]) {
    for assertion in assertions {
        match assertion {
            CaseAssertion::ExactResultHash(expected) => assert_exact_result_hash(case, expected),
            CaseAssertion::SchemaHash(expected) => assert_schema_hash(case, expected),
            CaseAssertion::ExpectedErrorContains(needle) => {
                assert_expected_error_contains(case, needle)
            }
            CaseAssertion::VersionMonotonicity => assert_version_monotonicity(case),
        }
    }
}

fn assert_exact_result_hash(case: &mut CaseResult, expected: &str) {
    if !case.success {
        return;
    }
    let found = case
        .samples
        .iter()
        .filter_map(|sample| sample.metrics.as_ref())
        .find_map(|metrics| metrics.result_hash.as_deref());
    if found != Some(expected) {
        fail_case(
            case,
            format!(
                "result hash mismatch: expected '{expected}', found '{}'",
                found.unwrap_or("none")
            ),
        );
    }
}

fn assert_schema_hash(case: &mut CaseResult, expected: &str) {
    if !case.success {
        return;
    }
    let found = case
        .samples
        .iter()
        .filter_map(|sample| sample.metrics.as_ref())
        .find_map(|metrics| metrics.schema_hash.as_deref());
    if found != Some(expected) {
        fail_case(
            case,
            format!(
                "schema hash mismatch: expected '{expected}', found '{}'",
                found.unwrap_or("none")
            ),
        );
    }
}

fn assert_expected_error_contains(case: &mut CaseResult, needle: &str) {
    let Some(failure) = case.failure.as_ref() else {
        fail_case(
            case,
            format!(
                "expected error assertion failed: expected case to fail with message containing '{needle}', but case succeeded"
            ),
        );
        return;
    };
    if failure.message.contains(needle) {
        case.success = true;
        case.classification = "expected_failure".to_string();
    }
}

fn assert_version_monotonicity(case: &mut CaseResult) {
    if !case.success {
        return;
    }
    let mut previous: Option<u64> = None;
    for version in case
        .samples
        .iter()
        .filter_map(|sample| sample.metrics.as_ref())
        .filter_map(|metrics| metrics.table_version)
    {
        if let Some(prev) = previous {
            if version < prev {
                fail_case(
                    case,
                    format!(
                        "version monotonicity assertion failed: table version decreased from {prev} to {version}"
                    ),
                );
                return;
            }
        }
        previous = Some(version);
    }
}

fn fail_case(case: &mut CaseResult, message: String) {
    case.success = false;
    case.failure = Some(CaseFailure { message });
}
