use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::assertions::CaseAssertion;
use crate::error::{BenchError, BenchResult};

pub const DEFAULT_RUST_MANIFEST_PATH: &str = "bench/manifests/core_rust.yaml";
pub const DEFAULT_PYTHON_MANIFEST_PATH: &str = "bench/manifests/core_python.yaml";

#[derive(Clone, Debug, Deserialize)]
pub struct BenchmarkManifest {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub cases: Vec<ManifestCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestCase {
    pub id: String,
    pub target: String,
    #[serde(default = "default_runner")]
    pub runner: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub assertions: Vec<ManifestAssertion>,
}

const fn default_enabled() -> bool {
    true
}

fn default_runner() -> String {
    "rust".to_string()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ManifestAssertion {
    ExactResultHash { value: String },
    SchemaHash { value: String },
    ExpectedErrorContains { value: String },
    VersionMonotonicity,
}

impl ManifestAssertion {
    pub fn to_case_assertion(&self) -> CaseAssertion {
        match self {
            Self::ExactResultHash { value } => CaseAssertion::ExactResultHash(value.clone()),
            Self::SchemaHash { value } => CaseAssertion::SchemaHash(value.clone()),
            Self::ExpectedErrorContains { value } => {
                CaseAssertion::ExpectedErrorContains(value.clone())
            }
            Self::VersionMonotonicity => CaseAssertion::VersionMonotonicity,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetId {
    TinySmoke,
    MediumSelective,
    SmallFiles,
    ManyVersions,
    TpcdsDuckdb,
}

impl DatasetId {
    pub fn parse(value: &str) -> BenchResult<Self> {
        match value {
            "tiny_smoke" => Ok(Self::TinySmoke),
            "medium_selective" => Ok(Self::MediumSelective),
            "small_files" => Ok(Self::SmallFiles),
            "many_versions" => Ok(Self::ManyVersions),
            "tpcds_duckdb" => Ok(Self::TpcdsDuckdb),
            other => Err(BenchError::InvalidArgument(format!(
                "unknown dataset_id '{other}' (expected one of: tiny_smoke, medium_selective, small_files, many_versions, tpcds_duckdb)"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TinySmoke => "tiny_smoke",
            Self::MediumSelective => "medium_selective",
            Self::SmallFiles => "small_files",
            Self::ManyVersions => "many_versions",
            Self::TpcdsDuckdb => "tpcds_duckdb",
        }
    }

    pub const fn scale(self) -> &'static str {
        match self {
            Self::TinySmoke => "sf1",
            Self::MediumSelective => "sf10",
            // P0 maps these scenario IDs to the currently supported scale; suites derive shape.
            Self::SmallFiles => "sf1",
            Self::ManyVersions => "sf1",
            Self::TpcdsDuckdb => "sf1",
        }
    }

    pub const fn fixture_profile(self) -> &'static str {
        match self {
            Self::ManyVersions => "many_versions",
            Self::TpcdsDuckdb => "tpcds_duckdb",
            Self::TinySmoke | Self::MediumSelective | Self::SmallFiles => "standard",
        }
    }
}

pub fn load_manifest(path: impl AsRef<Path>) -> BenchResult<BenchmarkManifest> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    serde_yaml::from_slice::<BenchmarkManifest>(&bytes).map_err(|error| {
        BenchError::InvalidArgument(format!("invalid manifest '{}': {error}", path.display()))
    })
}

pub fn benchmark_repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn ensure_required_manifests_exist() -> BenchResult<()> {
    ensure_required_manifests_exist_under_root(&benchmark_repo_root())
}

pub fn ensure_required_manifests_exist_under_root(root: &Path) -> BenchResult<()> {
    let required = [DEFAULT_RUST_MANIFEST_PATH, DEFAULT_PYTHON_MANIFEST_PATH];
    let mut missing = Vec::new();
    for relative in required {
        let path = root.join(relative);
        if !path.is_file() {
            missing.push((relative, path));
        }
    }
    if missing.is_empty() {
        return Ok(());
    }

    let details = missing
        .iter()
        .map(|(relative, path)| format!("- {relative} (expected at {})", path.display()))
        .collect::<Vec<_>>()
        .join("\n");

    Err(BenchError::InvalidArgument(format!(
        "manifest preflight failed for delta-bench `list`/`run` commands:\n{details}\n\
         ensure manifest files are present under `bench/manifests`."
    )))
}
