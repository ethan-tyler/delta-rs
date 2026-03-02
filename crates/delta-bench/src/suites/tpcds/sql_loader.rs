use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{BenchError, BenchResult};

use super::catalog::TpcdsQuerySpec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedTpcdsQuery {
    pub id: String,
    pub sql: String,
    pub path: PathBuf,
}

pub fn load_enabled_queries(specs: &[TpcdsQuerySpec]) -> BenchResult<Vec<LoadedTpcdsQuery>> {
    load_enabled_queries_from_dir(specs, &default_sql_dir())
}

pub fn load_enabled_queries_from_dir(
    specs: &[TpcdsQuerySpec],
    sql_dir: &Path,
) -> BenchResult<Vec<LoadedTpcdsQuery>> {
    let mut out = Vec::new();
    for spec in specs.iter().filter(|spec| spec.enabled) {
        let path = sql_dir.join(spec.sql_file);
        let sql = fs::read_to_string(&path).map_err(|err| {
            BenchError::InvalidArgument(format!(
                "failed to load SQL for query {} at {}: {}",
                spec.id,
                path.display(),
                err
            ))
        })?;
        out.push(LoadedTpcdsQuery {
            id: spec.id.to_string(),
            sql,
            path,
        });
    }
    Ok(out)
}

pub(crate) fn default_sql_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("suites")
        .join("tpcds")
        .join("sql")
}
