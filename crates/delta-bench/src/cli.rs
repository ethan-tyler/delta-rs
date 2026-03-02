use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::error::{BenchError, BenchResult};

#[derive(Debug, Parser)]
#[command(name = "delta-bench", about = "delta-rs macro benchmark harness")]
pub struct Args {
    #[arg(long, env = "DELTA_BENCH_FIXTURES", default_value = "fixtures")]
    pub fixtures_dir: PathBuf,
    #[arg(long, env = "DELTA_BENCH_RESULTS", default_value = "results")]
    pub results_dir: PathBuf,
    #[arg(long, env = "DELTA_BENCH_LABEL", default_value = "local")]
    pub label: String,
    #[arg(long)]
    pub git_sha: Option<String>,
    #[arg(
        long,
        env = "DELTA_BENCH_STORAGE_BACKEND",
        value_enum,
        default_value_t = StorageBackend::Local
    )]
    pub storage_backend: StorageBackend,
    #[arg(long = "storage-option")]
    pub storage_options: Vec<String>,
    #[arg(long, env = "DELTA_BENCH_BACKEND_PROFILE")]
    pub backend_profile: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StorageBackend {
    Local,
    S3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RunnerMode {
    Rust,
    Python,
    All,
}

impl RunnerMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    List {
        #[arg(default_value = "all")]
        target: String,
    },
    Data {
        #[arg(long, default_value = "sf1")]
        scale: String,
        #[arg(long)]
        dataset_id: Option<String>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        force: bool,
    },
    Run {
        #[arg(long, default_value = "sf1")]
        scale: String,
        #[arg(long)]
        dataset_id: Option<String>,
        #[arg(long, default_value = "all")]
        target: String,
        #[arg(long)]
        case_filter: Option<String>,
        #[arg(long, value_enum, default_value_t = RunnerMode::All)]
        runner: RunnerMode,
        #[arg(long, default_value_t = 1)]
        warmup: u32,
        #[arg(long, default_value_t = 5)]
        iterations: u32,
    },
    Doctor,
}

pub fn validate_label(label: &str) -> BenchResult<()> {
    if label.is_empty() {
        return Err(BenchError::InvalidArgument(
            "label must not be empty".to_string(),
        ));
    }
    if matches!(label, "." | "..") {
        return Err(BenchError::InvalidArgument(format!(
            "label '{label}' is not allowed"
        )));
    }
    if !label
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
    {
        return Err(BenchError::InvalidArgument(
            "label contains invalid characters; allowed: [A-Za-z0-9._-]".to_string(),
        ));
    }
    Ok(())
}

pub fn parse_storage_options(entries: &[String]) -> BenchResult<HashMap<String, String>> {
    let mut options = HashMap::new();
    for entry in entries {
        let Some((key, value)) = entry.split_once('=') else {
            return Err(BenchError::InvalidArgument(format!(
                "invalid storage option '{entry}'; expected KEY=VALUE"
            )));
        };
        if key.trim().is_empty() {
            return Err(BenchError::InvalidArgument(format!(
                "invalid storage option '{entry}'; key must not be empty"
            )));
        }
        options.insert(key.trim().to_string(), value.to_string());
    }
    Ok(options)
}
