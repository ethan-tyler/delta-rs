use std::time::Instant;
use std::{future::Future, time::Duration};

use crate::results::{CaseFailure, CaseResult, ElapsedStats, IterationSample, SampleMetrics};
use crate::stats::compute_stats;

#[derive(Clone, Debug)]
#[must_use]
pub enum CaseExecutionResult {
    Success(CaseResult),
    Failure(CaseResult),
}

pub fn run_case<F, M, E>(name: &str, warmup: u32, iterations: u32, mut op: F) -> CaseExecutionResult
where
    F: FnMut() -> Result<M, E>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        if let Err(error) = op() {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        match op() {
            Ok(metrics) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let metrics = metrics.into();
                samples.push(IterationSample {
                    elapsed_ms,
                    rows: metrics.rows_processed,
                    bytes: metrics.bytes_processed,
                    metrics: Some(metrics),
                });
            }
            Err(e) => {
                let failure = CaseFailure {
                    message: e.to_string(),
                };
                let case = failure_case_result(name, samples, failure.message);
                return CaseExecutionResult::Failure(case);
            }
        }
    }

    let case = success_case_result(name, samples);
    CaseExecutionResult::Success(case)
}

pub async fn run_case_async<F, Fut, M, E>(
    name: &str,
    warmup: u32,
    iterations: u32,
    mut op: F,
) -> CaseExecutionResult
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<M, E>>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        if let Err(error) = op().await {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        match op().await {
            Ok(metrics) => {
                append_sample(&mut samples, start.elapsed(), metrics, None);
            }
            Err(e) => {
                let case = failure_case_result(name, samples, e.to_string());
                return CaseExecutionResult::Failure(case);
            }
        }
    }

    CaseExecutionResult::Success(success_case_result(name, samples))
}

pub async fn run_case_async_custom_timing<F, Fut, M, E>(
    name: &str,
    warmup: u32,
    iterations: u32,
    mut op: F,
) -> CaseExecutionResult
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(M, Option<f64>), E>>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        if let Err(error) = op().await {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        match op().await {
            Ok((metrics, elapsed_ms_override)) => {
                append_sample(&mut samples, start.elapsed(), metrics, elapsed_ms_override);
            }
            Err(e) => {
                let case = failure_case_result(name, samples, e.to_string());
                return CaseExecutionResult::Failure(case);
            }
        }
    }

    CaseExecutionResult::Success(success_case_result(name, samples))
}

pub async fn run_case_async_with_setup<S, SetupF, F, Fut, M, E>(
    name: &str,
    warmup: u32,
    iterations: u32,
    mut setup: SetupF,
    mut op: F,
) -> CaseExecutionResult
where
    SetupF: FnMut() -> Result<S, E>,
    F: FnMut(S) -> Fut,
    Fut: Future<Output = Result<M, E>>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        let input = match setup() {
            Ok(input) => input,
            Err(error) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    Vec::new(),
                    format!(
                        "warmup setup iteration {} failed: {}",
                        warmup_idx + 1,
                        error.to_string()
                    ),
                ))
            }
        };
        if let Err(error) = op(input).await {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let input = match setup() {
            Ok(input) => input,
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        };

        let start = Instant::now();
        match op(input).await {
            Ok(metrics) => {
                append_sample(&mut samples, start.elapsed(), metrics, None);
            }
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        }
    }

    CaseExecutionResult::Success(success_case_result(name, samples))
}

pub async fn run_case_async_with_async_setup<S, SetupF, SetupFut, F, Fut, M, E>(
    name: &str,
    warmup: u32,
    iterations: u32,
    mut setup: SetupF,
    mut op: F,
) -> CaseExecutionResult
where
    SetupF: FnMut() -> SetupFut,
    SetupFut: Future<Output = Result<S, E>>,
    F: FnMut(S) -> Fut,
    Fut: Future<Output = Result<M, E>>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        let input = match setup().await {
            Ok(input) => input,
            Err(error) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    Vec::new(),
                    format!(
                        "warmup setup iteration {} failed: {}",
                        warmup_idx + 1,
                        error.to_string()
                    ),
                ))
            }
        };
        if let Err(error) = op(input).await {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let input = match setup().await {
            Ok(input) => input,
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        };

        let start = Instant::now();
        match op(input).await {
            Ok(metrics) => {
                append_sample(&mut samples, start.elapsed(), metrics, None);
            }
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        }
    }

    CaseExecutionResult::Success(success_case_result(name, samples))
}

pub async fn run_case_async_with_async_setup_custom_timing<S, SetupF, SetupFut, F, Fut, M, E>(
    name: &str,
    warmup: u32,
    iterations: u32,
    mut setup: SetupF,
    mut op: F,
) -> CaseExecutionResult
where
    SetupF: FnMut() -> SetupFut,
    SetupFut: Future<Output = Result<S, E>>,
    F: FnMut(S) -> Fut,
    Fut: Future<Output = Result<(M, Option<f64>), E>>,
    M: Into<SampleMetrics>,
    E: ToString,
{
    for warmup_idx in 0..warmup {
        let input = match setup().await {
            Ok(input) => input,
            Err(error) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    Vec::new(),
                    format!(
                        "warmup setup iteration {} failed: {}",
                        warmup_idx + 1,
                        error.to_string()
                    ),
                ))
            }
        };
        if let Err(error) = op(input).await {
            return CaseExecutionResult::Failure(failure_case_result(
                name,
                Vec::new(),
                format!(
                    "warmup iteration {} failed: {}",
                    warmup_idx + 1,
                    error.to_string()
                ),
            ));
        }
    }

    let mut samples = Vec::new();
    for _ in 0..iterations {
        let input = match setup().await {
            Ok(input) => input,
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        };

        let start = Instant::now();
        match op(input).await {
            Ok((metrics, elapsed_ms_override)) => {
                append_sample(&mut samples, start.elapsed(), metrics, elapsed_ms_override);
            }
            Err(e) => {
                return CaseExecutionResult::Failure(failure_case_result(
                    name,
                    samples,
                    e.to_string(),
                ))
            }
        }
    }

    CaseExecutionResult::Success(success_case_result(name, samples))
}

fn append_sample<M>(
    samples: &mut Vec<IterationSample>,
    elapsed: Duration,
    metrics: M,
    elapsed_ms_override: Option<f64>,
) where
    M: Into<SampleMetrics>,
{
    let metrics = metrics.into();
    samples.push(IterationSample {
        elapsed_ms: elapsed_ms_override.unwrap_or(elapsed.as_secs_f64() * 1000.0),
        rows: metrics.rows_processed,
        bytes: metrics.bytes_processed,
        metrics: Some(metrics),
    });
}

fn success_case_result(name: &str, samples: Vec<IterationSample>) -> CaseResult {
    CaseResult {
        case: name.to_string(),
        success: true,
        classification: "supported".to_string(),
        elapsed_stats: elapsed_stats_from_samples(&samples),
        samples,
        failure: None,
    }
}

fn failure_case_result(name: &str, samples: Vec<IterationSample>, message: String) -> CaseResult {
    CaseResult {
        case: name.to_string(),
        success: false,
        classification: "supported".to_string(),
        elapsed_stats: elapsed_stats_from_samples(&samples),
        samples,
        failure: Some(CaseFailure { message }),
    }
}

fn elapsed_stats_from_samples(samples: &[IterationSample]) -> Option<ElapsedStats> {
    let elapsed = samples
        .iter()
        .map(|sample| sample.elapsed_ms)
        .collect::<Vec<_>>();
    let stats = compute_stats(&elapsed)?;
    Some(ElapsedStats {
        min_ms: stats.min_ms,
        max_ms: stats.max_ms,
        mean_ms: stats.mean_ms,
        median_ms: stats.median_ms,
        stddev_ms: stats.stddev_ms,
        cv_pct: stats.cv_pct,
    })
}
