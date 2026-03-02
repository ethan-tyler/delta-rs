use delta_bench::assertions::{apply_case_assertions, CaseAssertion};
use delta_bench::results::{
    CaseFailure, CaseResult, IterationSample, RuntimeIOMetrics, SampleMetrics,
};

fn sample_with_hashes(
    result_hash: Option<&str>,
    schema_hash: Option<&str>,
    table_version: Option<u64>,
) -> IterationSample {
    IterationSample {
        elapsed_ms: 1.0,
        rows: Some(1),
        bytes: None,
        metrics: Some(
            SampleMetrics::base(Some(1), None, Some(1), table_version).with_runtime_io(
                RuntimeIOMetrics {
                    peak_rss_mb: None,
                    cpu_time_ms: None,
                    bytes_read: None,
                    bytes_written: None,
                    files_touched: None,
                    files_skipped: None,
                    spill_bytes: None,
                    result_hash: result_hash.map(ToOwned::to_owned),
                    schema_hash: schema_hash.map(ToOwned::to_owned),
                },
            ),
        ),
    }
}

fn case_result(
    success: bool,
    classification: &str,
    samples: Vec<IterationSample>,
    failure: Option<CaseFailure>,
) -> CaseResult {
    CaseResult {
        case: "test_case".to_string(),
        success,
        classification: classification.to_string(),
        samples,
        elapsed_stats: None,
        failure,
    }
}

#[test]
fn expected_error_assertion_reclassifies_failure() {
    let mut case = case_result(
        false,
        "supported",
        Vec::new(),
        Some(CaseFailure {
            message: "deletion vectors are not supported".to_string(),
        }),
    );

    apply_case_assertions(
        &mut case,
        &[CaseAssertion::ExpectedErrorContains(
            "not supported".to_string(),
        )],
    );

    assert!(case.success);
    assert_eq!(case.classification, "expected_failure");
}

#[test]
fn expected_error_assertion_fails_when_case_unexpectedly_succeeds() {
    let mut case = case_result(true, "supported", Vec::new(), None);

    apply_case_assertions(
        &mut case,
        &[CaseAssertion::ExpectedErrorContains(
            "not supported".to_string(),
        )],
    );

    assert!(!case.success);
    assert_eq!(case.classification, "supported");
    let message = case
        .failure
        .as_ref()
        .map(|f| f.message.as_str())
        .unwrap_or("");
    assert!(message.contains("expected error"));
}

#[test]
fn exact_result_hash_assertion_fails_mismatch() {
    let mut case = case_result(
        true,
        "supported",
        vec![sample_with_hashes(
            Some("sha256:result-abc"),
            Some("sha256:schema-abc"),
            Some(1),
        )],
        None,
    );

    apply_case_assertions(
        &mut case,
        &[CaseAssertion::ExactResultHash(
            "sha256:result-def".to_string(),
        )],
    );

    assert!(!case.success);
    let message = case
        .failure
        .as_ref()
        .map(|f| f.message.as_str())
        .unwrap_or("");
    assert!(message.contains("result hash mismatch"));
}

#[test]
fn schema_hash_assertion_uses_schema_hash_field_not_result_hash() {
    let mut case = case_result(
        true,
        "supported",
        vec![sample_with_hashes(
            Some("sha256:result-value"),
            Some("sha256:schema-value"),
            Some(1),
        )],
        None,
    );

    apply_case_assertions(
        &mut case,
        &[CaseAssertion::SchemaHash("sha256:schema-value".to_string())],
    );

    assert!(
        case.success,
        "schema hash assertion should pass when schema_hash matches"
    );
    assert!(case.failure.is_none());
}

#[test]
fn version_monotonicity_assertion_fails_on_decrease() {
    let mut case = case_result(
        true,
        "supported",
        vec![
            sample_with_hashes(Some("sha256:r"), Some("sha256:s"), Some(2)),
            sample_with_hashes(Some("sha256:r"), Some("sha256:s"), Some(1)),
        ],
        None,
    );

    apply_case_assertions(&mut case, &[CaseAssertion::VersionMonotonicity]);

    assert!(!case.success);
    let message = case
        .failure
        .as_ref()
        .map(|f| f.message.as_str())
        .unwrap_or("");
    assert!(message.contains("version monotonicity"));
}
