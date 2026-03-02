use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::assertions::{apply_case_assertions, CaseAssertion};
use crate::cli::RunnerMode;
use crate::error::{BenchError, BenchResult};
use crate::manifests::{load_manifest, DEFAULT_PYTHON_MANIFEST_PATH, DEFAULT_RUST_MANIFEST_PATH};
use crate::results::{CaseFailure, CaseResult};
use crate::runner::CaseExecutionResult;
use crate::storage::StorageConfig;

pub(crate) fn copy_dir_all(src: &Path, dst: &Path) -> BenchResult<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(BenchError::InvalidArgument(format!(
                "symlinks are not allowed in fixture tree: {}",
                entry.path().display()
            )));
        }
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}

pub(crate) fn into_case_result(result: CaseExecutionResult) -> CaseResult {
    match result {
        CaseExecutionResult::Success(c) | CaseExecutionResult::Failure(c) => c,
    }
}

pub(crate) fn fixture_error_cases(case_names: Vec<String>, message: &str) -> Vec<CaseResult> {
    case_names
        .into_iter()
        .map(|case| CaseResult {
            case,
            success: false,
            classification: "supported".to_string(),
            samples: Vec::new(),
            elapsed_stats: None,
            failure: Some(CaseFailure {
                message: format!("fixture load failed: {message}"),
            }),
        })
        .collect()
}

pub mod delete_update;
pub mod interop_py;
pub mod merge;
pub mod metadata;
pub mod optimize_vacuum;
pub mod scan;
mod scan_metrics;
pub mod tpcds;
pub mod write;

/// Single source of truth for suite names. Adding a new suite requires updating
/// this array, `list_cases_for_target`, and `run_target`.
const SUITE_NAMES: [&str; 8] = [
    "scan",
    "write",
    "delete_update",
    "merge",
    "metadata",
    "optimize_vacuum",
    "tpcds",
    "interop_py",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedCase {
    pub id: String,
    pub target: String,
    pub assertions: Vec<CaseAssertion>,
}

pub fn list_targets() -> Vec<&'static str> {
    let mut targets: Vec<&str> = SUITE_NAMES.to_vec();
    targets.push("all");
    targets
}

pub fn plan_run_cases(
    target: &str,
    runner: RunnerMode,
    case_filter: Option<&str>,
) -> BenchResult<Vec<PlannedCase>> {
    let canonical_target = canonical_suite_target(target);
    validate_runner_target(runner, canonical_target)?;
    let mut planned = plan_cases_from_manifest(canonical_target, runner)?;

    if let Some(filter) = case_filter.map(str::trim).filter(|value| !value.is_empty()) {
        planned.retain(|case| case.id.contains(filter));
    }
    if planned.is_empty() {
        return Err(BenchError::InvalidArgument(format!(
            "case filter matched no cases for target='{target}' (canonical='{canonical_target}') and runner='{}'",
            runner.as_str()
        )));
    }
    reject_duplicate_planned_case_ids(&planned)?;
    Ok(planned)
}

pub fn apply_dataset_assertion_policy(planned: &mut [PlannedCase], dataset_id: Option<&str>) {
    if dataset_id != Some("tpcds_duckdb") {
        return;
    }
    for case in planned.iter_mut().filter(|case| case.target == "tpcds") {
        case.assertions
            .retain(|assertion| !matches!(assertion, CaseAssertion::ExactResultHash(_)));
    }
}

pub async fn run_planned_cases(
    fixtures_dir: &Path,
    planned: &[PlannedCase],
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    let mut target_order = Vec::<String>::new();
    let mut seen_targets = HashSet::<String>::new();
    for case in planned {
        if seen_targets.insert(case.target.clone()) {
            target_order.push(case.target.clone());
        }
    }

    let mut by_target_and_case = HashMap::<(String, String), CaseResult>::new();
    for target in target_order {
        let target_results = run_target(
            fixtures_dir,
            target.as_str(),
            scale,
            warmup,
            iterations,
            storage,
        )
        .await?;
        for case in target_results {
            by_target_and_case.insert((target.clone(), case.case.clone()), case);
        }
    }

    let mut ordered = Vec::with_capacity(planned.len());
    for plan in planned {
        let key = (plan.target.clone(), plan.id.clone());
        let mut case = by_target_and_case.get(&key).cloned().ok_or_else(|| {
            BenchError::InvalidArgument(format!(
                "planned case '{}' for target '{}' was not produced by suite execution",
                plan.id, plan.target
            ))
        })?;
        if !plan.assertions.is_empty() {
            apply_case_assertions(&mut case, &plan.assertions);
        }
        ordered.push(case);
    }
    Ok(ordered)
}

pub fn list_cases_for_target(target: &str) -> BenchResult<Vec<String>> {
    let canonical_target = canonical_suite_target(target);
    match canonical_target {
        "scan" => Ok(scan::case_names()),
        "write" => Ok(write::case_names()),
        "delete_update" => Ok(delete_update::case_names()),
        "merge" => Ok(merge::case_names()),
        "metadata" => Ok(metadata::case_names()),
        "optimize_vacuum" => Ok(optimize_vacuum::case_names()),
        "tpcds" => Ok(tpcds::case_names()),
        "interop_py" => Ok(interop_py::case_names()),
        "all" => {
            let mut names = Vec::new();
            for suite in SUITE_NAMES {
                names.extend(list_cases_for_target(suite)?);
            }
            Ok(names)
        }
        other => Err(BenchError::InvalidArgument(format!(
            "unknown suite target: {other}"
        ))),
    }
}

fn canonical_suite_target(target: &str) -> &str {
    match target {
        "read_scan" => "scan",
        "delete_update_dml" => "delete_update",
        "merge_dml" => "merge",
        other => other,
    }
}

fn validate_runner_target(runner: RunnerMode, target: &str) -> BenchResult<()> {
    match runner {
        RunnerMode::Rust if target == "interop_py" => Err(BenchError::InvalidArgument(
            "runner=rust cannot run target=interop_py".to_string(),
        )),
        RunnerMode::Python if target != "all" && target != "interop_py" => {
            Err(BenchError::InvalidArgument(format!(
                "runner=python can only run target=interop_py or target=all (resolved target: {target})"
            )))
        }
        _ => Ok(()),
    }
}

fn plan_cases_from_manifest(target: &str, runner: RunnerMode) -> BenchResult<Vec<PlannedCase>> {
    plan_cases_from_manifest_paths(
        target,
        runner,
        DEFAULT_RUST_MANIFEST_PATH,
        DEFAULT_PYTHON_MANIFEST_PATH,
    )
}

fn plan_cases_from_manifest_paths(
    target: &str,
    runner: RunnerMode,
    rust_manifest_path: &str,
    python_manifest_path: &str,
) -> BenchResult<Vec<PlannedCase>> {
    let mut out = Vec::new();
    match runner {
        RunnerMode::Rust => {
            append_manifest_cases(&mut out, rust_manifest_path, target, "rust")?;
        }
        RunnerMode::Python => {
            append_manifest_cases(&mut out, python_manifest_path, target, "python")?;
        }
        RunnerMode::All => {
            append_manifest_cases(&mut out, rust_manifest_path, target, "rust")?;
            append_manifest_cases(&mut out, python_manifest_path, target, "python")?;
        }
    }
    Ok(out)
}

fn append_manifest_cases(
    out: &mut Vec<PlannedCase>,
    path: &str,
    target: &str,
    runner_name: &str,
) -> BenchResult<()> {
    let resolved_path = resolve_manifest_path(path);
    let display_path = resolved_path.display().to_string();
    let manifest = load_manifest(&resolved_path).map_err(|error| {
        BenchError::InvalidArgument(format!(
            "failed to load required manifest '{display_path}': {error}"
        ))
    })?;

    for case in manifest.cases {
        if !case.enabled {
            continue;
        }
        if case.runner != runner_name {
            continue;
        }
        if target != "all" && case.target != target {
            continue;
        }
        out.push(PlannedCase {
            id: case.id,
            target: case.target,
            assertions: case
                .assertions
                .iter()
                .map(|assertion| assertion.to_case_assertion())
                .collect(),
        });
    }
    Ok(())
}

fn resolve_manifest_path(path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return candidate.to_path_buf();
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(candidate)
}

fn reject_duplicate_planned_case_ids(planned: &[PlannedCase]) -> BenchResult<()> {
    let mut seen = HashSet::new();
    for case in planned {
        if !seen.insert(case.id.as_str()) {
            return Err(BenchError::InvalidArgument(format!(
                "planned case list contains duplicate case id '{}'",
                case.id
            )));
        }
    }
    Ok(())
}

async fn run_single_suite(
    fixtures_dir: &Path,
    suite: &str,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    match suite {
        "scan" => scan::run(fixtures_dir, scale, warmup, iterations, storage).await,
        "write" => write::run(fixtures_dir, scale, warmup, iterations, storage).await,
        "delete_update" => {
            delete_update::run(fixtures_dir, scale, warmup, iterations, storage).await
        }
        "merge" => merge::run(fixtures_dir, scale, warmup, iterations, storage).await,
        "metadata" => metadata::run(fixtures_dir, scale, warmup, iterations, storage).await,
        "optimize_vacuum" => {
            optimize_vacuum::run(fixtures_dir, scale, warmup, iterations, storage).await
        }
        "tpcds" => tpcds::run(fixtures_dir, scale, warmup, iterations, storage).await,
        "interop_py" => interop_py::run(fixtures_dir, scale, warmup, iterations, storage).await,
        other => Err(BenchError::InvalidArgument(format!(
            "unknown suite target: {other}"
        ))),
    }
}

pub async fn run_target(
    fixtures_dir: &Path,
    target: &str,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    let canonical_target = canonical_suite_target(target);
    if canonical_target == "all" {
        return Err(BenchError::InvalidArgument(
            "target=all requires manifest planning; use plan_run_cases + run_planned_cases"
                .to_string(),
        ));
    }
    run_single_suite(
        fixtures_dir,
        canonical_target,
        scale,
        warmup,
        iterations,
        storage,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{plan_cases_from_manifest_paths, RunnerMode};

    #[test]
    fn manifest_planning_fails_when_required_manifest_is_missing() {
        let err = plan_cases_from_manifest_paths(
            "all",
            RunnerMode::Rust,
            "/tmp/definitely-missing-rust-manifest.yaml",
            "/tmp/definitely-missing-python-manifest.yaml",
        )
        .expect_err("missing manifest should fail");
        assert!(
            err.to_string().contains("failed to load required manifest"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn manifest_planning_fails_when_manifest_is_invalid() {
        let temp = tempfile::tempdir().expect("tempdir");
        let rust_manifest = temp.path().join("rust.yaml");
        let python_manifest = temp.path().join("python.yaml");
        fs::write(&rust_manifest, "not: [valid").expect("write invalid rust manifest");
        fs::write(
            &python_manifest,
            "id: core-python\ndescription: test\ncases: []\n",
        )
        .expect("write valid python manifest");

        let err = plan_cases_from_manifest_paths(
            "all",
            RunnerMode::Rust,
            rust_manifest.to_str().expect("utf8 path"),
            python_manifest.to_str().expect("utf8 path"),
        )
        .expect_err("invalid manifest should fail");
        assert!(
            err.to_string().contains("invalid manifest"),
            "unexpected error: {err}"
        );
    }
}
