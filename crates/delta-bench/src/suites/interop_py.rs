use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;

use crate::error::{BenchError, BenchResult};
use crate::results::{
    validate_case_classification, CaseFailure, CaseResult, ElapsedStats, IterationSample,
    RuntimeIOMetrics, SampleMetrics,
};
use crate::stats::compute_stats;
use crate::storage::StorageConfig;

const CASES: [&str; 3] = [
    "pandas_roundtrip_smoke",
    "polars_roundtrip_smoke",
    "pyarrow_dataset_scan_perf",
];
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_RETRIES: u32 = 1;

#[derive(Debug, Deserialize)]
struct InteropCaseOutput {
    #[serde(default)]
    rows_processed: Option<u64>,
    #[serde(default)]
    bytes_processed: Option<u64>,
    #[serde(default)]
    operations: Option<u64>,
    #[serde(default)]
    table_version: Option<u64>,
    #[serde(default)]
    peak_rss_mb: Option<u64>,
    #[serde(default)]
    cpu_time_ms: Option<u64>,
    #[serde(default)]
    bytes_read: Option<u64>,
    #[serde(default)]
    bytes_written: Option<u64>,
    #[serde(default)]
    files_touched: Option<u64>,
    #[serde(default)]
    files_skipped: Option<u64>,
    #[serde(default)]
    spill_bytes: Option<u64>,
    #[serde(default)]
    result_hash: Option<String>,
    #[serde(default)]
    schema_hash: Option<String>,
    classification: String,
}

#[derive(Clone, Debug)]
struct InteropRuntimeConfig {
    timeout: Duration,
    retries: u32,
    python_executable: String,
}

impl InteropRuntimeConfig {
    fn from_env() -> BenchResult<Self> {
        let timeout_ms = parse_env_u64("DELTA_BENCH_INTEROP_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)?;
        if timeout_ms == 0 {
            return Err(BenchError::InvalidArgument(
                "DELTA_BENCH_INTEROP_TIMEOUT_MS must be > 0".to_string(),
            ));
        }
        let retries = parse_env_u64("DELTA_BENCH_INTEROP_RETRIES", DEFAULT_RETRIES as u64)?;
        if retries > u32::MAX as u64 {
            return Err(BenchError::InvalidArgument(format!(
                "DELTA_BENCH_INTEROP_RETRIES is too large: {retries}"
            )));
        }

        let python_executable = std::env::var("DELTA_BENCH_INTEROP_PYTHON")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "python3".to_string());

        Ok(Self {
            timeout: Duration::from_millis(timeout_ms),
            retries: retries as u32,
            python_executable,
        })
    }
}

fn parse_env_u64(name: &str, default: u64) -> BenchResult<u64> {
    let Some(raw) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(default);
    }
    trimmed.parse::<u64>().map_err(|error| {
        BenchError::InvalidArgument(format!("invalid value for {name}='{trimmed}': {error}"))
    })
}

pub fn case_names() -> Vec<String> {
    CASES.iter().map(|case| (*case).to_string()).collect()
}

pub async fn run(
    fixtures_dir: &Path,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    if !storage.is_local() {
        return Ok(case_names()
            .into_iter()
            .map(|case| CaseResult {
                case,
                success: true,
                classification: "expected_failure".to_string(),
                samples: Vec::new(),
                elapsed_stats: None,
                failure: Some(CaseFailure {
                    message: "interop_py currently supports local backend only in P0".to_string(),
                }),
            })
            .collect());
    }

    let runtime = InteropRuntimeConfig::from_env()?;
    let mut out = Vec::new();
    for case in CASES {
        out.push(run_case(case, fixtures_dir, scale, warmup, iterations, &runtime).await?);
    }
    Ok(out)
}

async fn run_case(
    case: &str,
    fixtures_dir: &Path,
    scale: &str,
    warmup: u32,
    iterations: u32,
    runtime: &InteropRuntimeConfig,
) -> BenchResult<CaseResult> {
    for _ in 0..warmup {
        let _ = run_python_case_with_runtime(case, fixtures_dir, scale, runtime, None).await;
    }

    let mut samples = Vec::new();
    let mut classification = "supported".to_string();

    for _ in 0..iterations {
        let started = Instant::now();
        match run_python_case_with_runtime(case, fixtures_dir, scale, runtime, None).await {
            Ok(output) => {
                classification = output.classification.clone();
                let metrics = SampleMetrics::base(
                    output.rows_processed,
                    output.bytes_processed,
                    output.operations,
                    output.table_version,
                )
                .with_runtime_io(RuntimeIOMetrics {
                    peak_rss_mb: output.peak_rss_mb,
                    cpu_time_ms: output.cpu_time_ms,
                    bytes_read: output.bytes_read,
                    bytes_written: output.bytes_written,
                    files_touched: output.files_touched,
                    files_skipped: output.files_skipped,
                    spill_bytes: output.spill_bytes,
                    result_hash: output.result_hash,
                    schema_hash: output.schema_hash,
                });
                samples.push(IterationSample {
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    rows: metrics.rows_processed,
                    bytes: metrics.bytes_processed,
                    metrics: Some(metrics),
                });
            }
            Err(error) => {
                return Ok(CaseResult {
                    case: case.to_string(),
                    success: false,
                    classification,
                    elapsed_stats: elapsed_stats_from_samples(&samples),
                    samples,
                    failure: Some(CaseFailure {
                        message: error.to_string(),
                    }),
                });
            }
        }
    }

    Ok(CaseResult {
        case: case.to_string(),
        success: true,
        classification,
        elapsed_stats: elapsed_stats_from_samples(&samples),
        samples,
        failure: None,
    })
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

async fn run_python_case_with_runtime(
    case: &str,
    fixtures_dir: &Path,
    scale: &str,
    runtime: &InteropRuntimeConfig,
    script_override: Option<&Path>,
) -> BenchResult<InteropCaseOutput> {
    let script = match script_override {
        Some(path) => path.to_path_buf(),
        None => {
            let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            repo_root
                .join("python")
                .join("delta_bench_interop")
                .join("run_case.py")
        }
    };

    let max_attempts = runtime.retries.saturating_add(1);
    for attempt in 1..=max_attempts {
        match run_python_case_once(case, fixtures_dir, scale, runtime, &script).await {
            Ok(output) => return Ok(output),
            Err(_error) if attempt < max_attempts => continue,
            Err(error) => {
                return Err(BenchError::InvalidArgument(format!(
                    "interop case '{case}' failed after {max_attempts} attempt(s): {error}"
                )));
            }
        }
    }

    Err(BenchError::InvalidArgument(format!(
        "interop case '{case}' did not execute any attempts"
    )))
}

async fn run_python_case_once(
    case: &str,
    fixtures_dir: &Path,
    scale: &str,
    runtime: &InteropRuntimeConfig,
    script: &Path,
) -> BenchResult<InteropCaseOutput> {
    let mut command = tokio::process::Command::new(&runtime.python_executable);
    command.kill_on_drop(true);
    command
        .arg(script)
        .arg("--case")
        .arg(case)
        .arg("--fixtures-dir")
        .arg(fixtures_dir)
        .arg("--scale")
        .arg(scale);
    let output = match tokio::time::timeout(runtime.timeout, command.output()).await {
        Ok(result) => result?,
        Err(_) => {
            return Err(BenchError::InvalidArgument(format!(
                "interop case '{case}' timed out after {} ms",
                runtime.timeout.as_millis()
            )));
        }
    };

    if !output.status.success() {
        return Err(BenchError::InvalidArgument(format!(
            "interop case '{case}' failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let parsed = serde_json::from_slice::<InteropCaseOutput>(&output.stdout).map_err(|error| {
        BenchError::InvalidArgument(format!(
            "failed to parse interop output for case '{case}': {error}"
        ))
    })?;
    validate_case_classification(parsed.classification.as_str()).map_err(|error| {
        BenchError::InvalidArgument(format!(
            "failed to parse interop output for case '{case}': {error}"
        ))
    })?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use super::{run_python_case_with_runtime, InteropRuntimeConfig};

    #[tokio::test]
    async fn python_runtime_enforces_timeout() {
        let temp = tempfile::tempdir().expect("tempdir");
        let script = temp.path().join("sleep_case.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import time
time.sleep(0.25)
print('{"rows_processed":1,"bytes_processed":1,"operations":1,"classification":"supported"}')
"#,
        )
        .expect("write script");

        let runtime = InteropRuntimeConfig {
            timeout: Duration::from_millis(10),
            retries: 0,
            python_executable: "python3".to_string(),
        };
        let err = run_python_case_with_runtime(
            "timeout_case",
            temp.path(),
            "sf1",
            &runtime,
            Some(script.as_path()),
        )
        .await
        .expect_err("slow script should time out");
        assert!(
            err.to_string().contains("timed out"),
            "unexpected timeout error: {err}"
        );
    }

    #[tokio::test]
    async fn python_runtime_retries_transient_failure() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state_file = temp.path().join("retry_state.txt");
        let script = temp.path().join("retry_case.py");
        fs::write(
            &script,
            format!(
                r#"#!/usr/bin/env python3
from pathlib import Path
state = Path(r"{state}")
if not state.exists():
    state.write_text("1", encoding="utf-8")
    raise SystemExit("first attempt fails")
print('{{"rows_processed":1,"bytes_processed":1,"operations":1,"classification":"supported"}}')
"#,
                state = state_file.display()
            ),
        )
        .expect("write script");

        let runtime = InteropRuntimeConfig {
            timeout: Duration::from_secs(1),
            retries: 1,
            python_executable: "python3".to_string(),
        };
        let out = run_python_case_with_runtime(
            "retry_case",
            temp.path(),
            "sf1",
            &runtime,
            Some(script.as_path()),
        )
        .await
        .expect("one retry should recover");
        assert_eq!(out.classification, "supported");
    }

    #[tokio::test]
    async fn python_runtime_rejects_invalid_classification() {
        let temp = tempfile::tempdir().expect("tempdir");
        let script = temp.path().join("bad_classification.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
print('{"rows_processed":1,"bytes_processed":1,"operations":1,"classification":"experimental"}')
"#,
        )
        .expect("write script");

        let runtime = InteropRuntimeConfig {
            timeout: Duration::from_secs(1),
            retries: 0,
            python_executable: "python3".to_string(),
        };
        let err = run_python_case_with_runtime(
            "bad_classification",
            temp.path(),
            "sf1",
            &runtime,
            Some(script.as_path()),
        )
        .await
        .expect_err("invalid classification should fail parsing");
        assert!(
            err.to_string().contains("classification"),
            "unexpected error: {err}"
        );
    }
}
