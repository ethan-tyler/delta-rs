use std::thread;
use std::time::Duration;

use delta_bench::runner::{
    run_case, run_case_async, run_case_async_custom_timing, run_case_async_with_async_setup,
    run_case_async_with_async_setup_custom_timing, run_case_async_with_setup, CaseExecutionResult,
};

#[tokio::test]
async fn setup_delay_is_not_counted_in_iteration_elapsed_time() {
    let result = run_case_async_with_setup(
        "timing_case",
        0,
        1,
        || -> Result<(), String> {
            thread::sleep(Duration::from_millis(25));
            Ok(())
        },
        |_| async { Ok::<u64, String>(1) },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => case,
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    };
    assert_eq!(case.samples.len(), 1);
    assert!(
        case.samples[0].elapsed_ms < 50.0,
        "setup delay leaked into measured time: {} ms (expected < 50 ms)",
        case.samples[0].elapsed_ms
    );
}

#[tokio::test]
async fn async_setup_delay_is_not_counted_in_iteration_elapsed_time() {
    let result = run_case_async_with_async_setup(
        "timing_case_async_setup",
        0,
        1,
        || async {
            tokio::time::sleep(Duration::from_millis(25)).await;
            Ok::<(), String>(())
        },
        |_| async { Ok::<u64, String>(1) },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => case,
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    };
    assert_eq!(case.samples.len(), 1);
    assert!(
        case.samples[0].elapsed_ms < 50.0,
        "async setup delay leaked into measured time: {} ms (expected < 50 ms)",
        case.samples[0].elapsed_ms
    );
}

#[tokio::test]
async fn successful_case_includes_elapsed_stats() {
    let result = run_case_async_with_setup(
        "timing_case_stats",
        0,
        3,
        || -> Result<(), String> { Ok(()) },
        |_| async {
            tokio::time::sleep(Duration::from_millis(2)).await;
            Ok::<u64, String>(1)
        },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => case,
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    };

    let elapsed_stats = case
        .elapsed_stats
        .as_ref()
        .expect("successful case should include elapsed stats");
    assert!(elapsed_stats.max_ms >= elapsed_stats.min_ms);
    assert!(elapsed_stats.stddev_ms >= 0.0);
}

#[test]
fn partial_failure_case_includes_elapsed_stats_for_collected_samples() {
    let mut attempts = 0_u32;
    let result = run_case("timing_case_failure_stats", 0, 3, || {
        attempts += 1;
        if attempts < 3 {
            Ok::<u64, &'static str>(1)
        } else {
            Err::<u64, &'static str>("boom")
        }
    });

    let case = match result {
        CaseExecutionResult::Success(case) => panic!("expected failure, got success: {:?}", case),
        CaseExecutionResult::Failure(case) => case,
    };
    assert_eq!(case.samples.len(), 2);
    assert!(
        case.elapsed_stats.is_some(),
        "failure case should retain elapsed stats for collected samples"
    );
}

#[test]
fn warmup_failure_in_sync_runner_is_reported_as_case_failure() {
    let mut attempts = 0_u32;
    let result = run_case("warmup_sync_failure", 1, 1, || {
        attempts += 1;
        if attempts == 1 {
            Err::<u64, &'static str>("warmup boom")
        } else {
            Ok::<u64, &'static str>(1)
        }
    });

    let case = match result {
        CaseExecutionResult::Success(case) => panic!("expected failure, got success: {:?}", case),
        CaseExecutionResult::Failure(case) => case,
    };
    let message = case
        .failure
        .as_ref()
        .expect("warmup failure should include failure payload")
        .message
        .to_ascii_lowercase();
    assert!(
        message.contains("warmup"),
        "failure should mention warmup phase, got: {message}"
    );
    assert_eq!(
        case.samples.len(),
        0,
        "warmup failures should not record timed iteration samples"
    );
}

#[tokio::test]
async fn warmup_failure_in_async_runner_is_reported_as_case_failure() {
    let mut attempts = 0_u32;
    let result = run_case_async("warmup_async_failure", 1, 1, || {
        attempts += 1;
        async move {
            if attempts == 1 {
                Err::<u64, &'static str>("warmup async boom")
            } else {
                Ok::<u64, &'static str>(1)
            }
        }
    })
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => panic!("expected failure, got success: {:?}", case),
        CaseExecutionResult::Failure(case) => case,
    };
    let message = case
        .failure
        .as_ref()
        .expect("warmup failure should include failure payload")
        .message
        .to_ascii_lowercase();
    assert!(
        message.contains("warmup"),
        "failure should mention warmup phase, got: {message}"
    );
}

#[tokio::test]
async fn warmup_setup_failure_in_async_setup_runner_is_reported_as_case_failure() {
    let mut attempts = 0_u32;
    let result = run_case_async_with_setup(
        "warmup_setup_failure",
        1,
        1,
        || {
            attempts += 1;
            if attempts == 1 {
                Err::<(), &'static str>("warmup setup boom")
            } else {
                Ok::<(), &'static str>(())
            }
        },
        |_| async { Ok::<u64, &'static str>(1) },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => panic!("expected failure, got success: {:?}", case),
        CaseExecutionResult::Failure(case) => case,
    };
    let message = case
        .failure
        .as_ref()
        .expect("warmup setup failure should include failure payload")
        .message
        .to_ascii_lowercase();
    assert!(
        message.contains("warmup"),
        "failure should mention warmup phase, got: {message}"
    );
}

#[tokio::test]
async fn warmup_setup_failure_in_async_async_setup_runner_is_reported_as_case_failure() {
    let mut attempts = 0_u32;
    let result = run_case_async_with_async_setup(
        "warmup_async_setup_failure",
        1,
        1,
        || {
            attempts += 1;
            async move {
                if attempts == 1 {
                    Err::<(), &'static str>("warmup async setup boom")
                } else {
                    Ok::<(), &'static str>(())
                }
            }
        },
        |_| async { Ok::<u64, &'static str>(1) },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => panic!("expected failure, got success: {:?}", case),
        CaseExecutionResult::Failure(case) => case,
    };
    let message = case
        .failure
        .as_ref()
        .expect("warmup async setup failure should include failure payload")
        .message
        .to_ascii_lowercase();
    assert!(
        message.contains("warmup"),
        "failure should mention warmup phase, got: {message}"
    );
}

#[tokio::test]
async fn custom_timing_override_controls_elapsed_for_async_case() {
    let result = run_case_async_custom_timing("timing_override_async", 0, 1, || async {
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok::<(u64, Option<f64>), String>((1, Some(1.25)))
    })
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => case,
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    };
    assert_eq!(case.samples.len(), 1);
    assert!(
        (case.samples[0].elapsed_ms - 1.25).abs() < 0.001,
        "expected override elapsed value 1.25 ms, got {} ms",
        case.samples[0].elapsed_ms
    );
}

#[tokio::test]
async fn custom_timing_override_controls_elapsed_for_async_setup_case() {
    let result = run_case_async_with_async_setup_custom_timing(
        "timing_override_async_setup",
        0,
        1,
        || async { Ok::<(), String>(()) },
        |_| async {
            tokio::time::sleep(Duration::from_millis(25)).await;
            Ok::<(u64, Option<f64>), String>((1, Some(2.5)))
        },
    )
    .await;

    let case = match result {
        CaseExecutionResult::Success(case) => case,
        CaseExecutionResult::Failure(case) => panic!("unexpected failure: {:?}", case.failure),
    };
    assert_eq!(case.samples.len(), 1);
    assert!(
        (case.samples[0].elapsed_ms - 2.5).abs() < 0.001,
        "expected override elapsed value 2.5 ms, got {} ms",
        case.samples[0].elapsed_ms
    );
}
