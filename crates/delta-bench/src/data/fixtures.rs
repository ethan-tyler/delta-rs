use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use deltalake_core::arrow;
use deltalake_core::protocol::SaveMode;
use url::Url;

use super::datasets::{FixtureManifest, NarrowSaleRow};
use super::generator::generate_narrow_sales_rows;
use crate::error::{BenchError, BenchResult};
use crate::storage::StorageConfig;

const NARROW_SALES_TABLE_DIR: &str = "narrow_sales_delta";
const MERGE_TARGET_TABLE_DIR: &str = "merge_target_delta";
const READ_PARTITIONED_TABLE_DIR: &str = "read_partitioned_delta";
const DELETE_UPDATE_SMALL_FILES_TABLE_DIR: &str = "delete_update_small_files_delta";
const MERGE_PARTITIONED_TARGET_TABLE_DIR: &str = "merge_partitioned_target_delta";
const OPTIMIZE_SMALL_FILES_TABLE_DIR: &str = "optimize_small_files_delta";
const OPTIMIZE_COMPACTED_TABLE_DIR: &str = "optimize_compacted_delta";
const VACUUM_READY_TABLE_DIR: &str = "vacuum_ready_delta";
const TPCDS_DIR: &str = "tpcds";
const TPCDS_STORE_SALES_TABLE_DIR: &str = "store_sales";
const FIXTURE_SCHEMA_VERSION: u32 = 2;
const MANY_VERSIONS_APPEND_COMMITS: usize = 12;
const FIXTURE_LOCK_DIR: &str = ".delta_bench_locks";
const DEFAULT_FIXTURE_LOCK_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_FIXTURE_LOCK_RETRY_MS: u64 = 50;
const FIXTURE_LOCK_TIMEOUT_ENV: &str = "DELTA_BENCH_FIXTURE_LOCK_TIMEOUT_MS";
const FIXTURE_LOCK_RETRY_ENV: &str = "DELTA_BENCH_FIXTURE_LOCK_RETRY_MS";
const DEFAULT_TPCDS_DUCKDB_TIMEOUT_MS: u64 = 600_000;
const TPCDS_DUCKDB_CHUNK_ROWS: usize = 10_000;
const TPCDS_DUCKDB_PYTHON_ENV: &str = "DELTA_BENCH_DUCKDB_PYTHON";
const TPCDS_DUCKDB_SCRIPT_ENV: &str = "DELTA_BENCH_TPCDS_DUCKDB_SCRIPT";
const TPCDS_DUCKDB_TIMEOUT_ENV: &str = "DELTA_BENCH_TPCDS_DUCKDB_TIMEOUT_MS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixtureProfile {
    Standard,
    ManyVersions,
    TpcdsDuckdb,
}

impl FixtureProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::ManyVersions => "many_versions",
            Self::TpcdsDuckdb => "tpcds_duckdb",
        }
    }
}

pub fn scale_to_row_count(scale: &str) -> BenchResult<usize> {
    match scale {
        "sf1" => Ok(10_000),
        "sf10" => Ok(100_000),
        "sf100" => Ok(1_000_000),
        _ => Err(BenchError::InvalidArgument(format!(
            "unknown scale '{scale}' (expected one of: sf1, sf10, sf100)"
        ))),
    }
}

pub fn validate_scale(scale: &str) -> BenchResult<()> {
    if scale.is_empty() {
        return Err(BenchError::InvalidArgument(
            "scale must not be empty".to_string(),
        ));
    }
    if matches!(scale, "." | "..") {
        return Err(BenchError::InvalidArgument(format!(
            "scale '{scale}' is not allowed"
        )));
    }
    if !scale
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
    {
        return Err(BenchError::InvalidArgument(
            "scale contains invalid characters; allowed: [A-Za-z0-9._-]".to_string(),
        ));
    }
    let _ = scale_to_row_count(scale)?;
    Ok(())
}

pub fn fixture_root(fixtures_dir: &Path, scale: &str) -> BenchResult<PathBuf> {
    Ok(fixtures_dir.join(scale))
}

pub fn narrow_sales_table_path(fixtures_dir: &Path, scale: &str) -> BenchResult<PathBuf> {
    Ok(fixture_root(fixtures_dir, scale)?.join(NARROW_SALES_TABLE_DIR))
}

pub fn merge_target_table_path(fixtures_dir: &Path, scale: &str) -> BenchResult<PathBuf> {
    Ok(fixture_root(fixtures_dir, scale)?.join(MERGE_TARGET_TABLE_DIR))
}

pub fn read_partitioned_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(READ_PARTITIONED_TABLE_DIR)
}

pub fn merge_partitioned_target_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(MERGE_PARTITIONED_TARGET_TABLE_DIR)
}

pub fn delete_update_small_files_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(DELETE_UPDATE_SMALL_FILES_TABLE_DIR)
}

pub fn optimize_small_files_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(OPTIMIZE_SMALL_FILES_TABLE_DIR)
}

pub fn optimize_compacted_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(OPTIMIZE_COMPACTED_TABLE_DIR)
}

pub fn vacuum_ready_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(VACUUM_READY_TABLE_DIR)
}

pub fn tpcds_store_sales_table_path(fixtures_dir: &Path, scale: &str) -> PathBuf {
    fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join(TPCDS_DIR)
        .join(TPCDS_STORE_SALES_TABLE_DIR)
}

fn required_local_fixture_tables_exist(root: &Path) -> bool {
    [
        NARROW_SALES_TABLE_DIR,
        MERGE_TARGET_TABLE_DIR,
        READ_PARTITIONED_TABLE_DIR,
        MERGE_PARTITIONED_TARGET_TABLE_DIR,
        OPTIMIZE_SMALL_FILES_TABLE_DIR,
        OPTIMIZE_COMPACTED_TABLE_DIR,
        VACUUM_READY_TABLE_DIR,
        "tpcds/store_sales",
    ]
    .iter()
    .all(|table| root.join(table).join("_delta_log").exists())
}

pub fn narrow_sales_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &narrow_sales_table_path(fixtures_dir, scale)?,
        scale,
        NARROW_SALES_TABLE_DIR,
    )
}

pub fn merge_target_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &merge_target_table_path(fixtures_dir, scale)?,
        scale,
        MERGE_TARGET_TABLE_DIR,
    )
}

pub fn read_partitioned_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &read_partitioned_table_path(fixtures_dir, scale),
        scale,
        READ_PARTITIONED_TABLE_DIR,
    )
}

pub fn merge_partitioned_target_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &merge_partitioned_target_table_path(fixtures_dir, scale),
        scale,
        MERGE_PARTITIONED_TARGET_TABLE_DIR,
    )
}

pub fn delete_update_small_files_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &delete_update_small_files_table_path(fixtures_dir, scale),
        scale,
        DELETE_UPDATE_SMALL_FILES_TABLE_DIR,
    )
}

pub fn optimize_small_files_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &optimize_small_files_table_path(fixtures_dir, scale),
        scale,
        OPTIMIZE_SMALL_FILES_TABLE_DIR,
    )
}

pub fn optimize_compacted_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &optimize_compacted_table_path(fixtures_dir, scale),
        scale,
        OPTIMIZE_COMPACTED_TABLE_DIR,
    )
}

pub fn vacuum_ready_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &vacuum_ready_table_path(fixtures_dir, scale),
        scale,
        VACUUM_READY_TABLE_DIR,
    )
}

pub fn tpcds_store_sales_table_url(
    fixtures_dir: &Path,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<Url> {
    storage.table_url_for(
        &tpcds_store_sales_table_path(fixtures_dir, scale),
        scale,
        "tpcds/store_sales",
    )
}

#[derive(Clone, Debug)]
struct TpcdsDuckdbRuntime {
    python_executable: String,
    script_path: PathBuf,
    timeout: Duration,
}

#[derive(Clone, Copy, Debug)]
struct TpcdsStoreSalesRow {
    ss_customer_sk: i64,
    ss_ext_sales_price: f64,
    ss_item_sk: i64,
    ss_quantity: i64,
    ss_sold_date_sk: i64,
}

struct FixtureGenerationLock {
    path: PathBuf,
}

impl Drop for FixtureGenerationLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl TpcdsDuckdbRuntime {
    fn from_env() -> BenchResult<Self> {
        let timeout_ms = parse_env_u64(TPCDS_DUCKDB_TIMEOUT_ENV, DEFAULT_TPCDS_DUCKDB_TIMEOUT_MS)?;
        if timeout_ms == 0 {
            return Err(BenchError::InvalidArgument(format!(
                "{TPCDS_DUCKDB_TIMEOUT_ENV} must be > 0"
            )));
        }
        let python_executable = std::env::var(TPCDS_DUCKDB_PYTHON_ENV)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "python3".to_string());
        let script_path = std::env::var(TPCDS_DUCKDB_SCRIPT_ENV)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_tpcds_duckdb_script_path);

        if !script_path.exists() {
            return Err(BenchError::InvalidArgument(format!(
                "duckdb generator script not found at '{}' (set {TPCDS_DUCKDB_SCRIPT_ENV} to override)",
                script_path.display()
            )));
        }

        Ok(Self {
            python_executable,
            script_path,
            timeout: Duration::from_millis(timeout_ms),
        })
    }
}

fn default_tpcds_duckdb_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("python")
        .join("delta_bench_tpcds")
        .join("generate_store_sales_csv.py")
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

async fn acquire_fixture_generation_lock(
    fixtures_dir: &Path,
    scale: &str,
) -> BenchResult<FixtureGenerationLock> {
    let timeout_ms = parse_env_u64(FIXTURE_LOCK_TIMEOUT_ENV, DEFAULT_FIXTURE_LOCK_TIMEOUT_MS)?;
    if timeout_ms == 0 {
        return Err(BenchError::InvalidArgument(format!(
            "{FIXTURE_LOCK_TIMEOUT_ENV} must be > 0"
        )));
    }
    let retry_ms = parse_env_u64(FIXTURE_LOCK_RETRY_ENV, DEFAULT_FIXTURE_LOCK_RETRY_MS)?;
    if retry_ms == 0 {
        return Err(BenchError::InvalidArgument(format!(
            "{FIXTURE_LOCK_RETRY_ENV} must be > 0"
        )));
    }

    let locks_root = fixtures_dir.join(FIXTURE_LOCK_DIR);
    fs::create_dir_all(&locks_root)?;
    let lock_path = locks_root.join(format!("{scale}.lock"));
    let start = std::time::Instant::now();

    loop {
        match fs::create_dir(&lock_path) {
            Ok(()) => return Ok(FixtureGenerationLock { path: lock_path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if start.elapsed().as_millis() >= u128::from(timeout_ms) {
                    return Err(BenchError::InvalidArgument(format!(
                        "timed out waiting for fixture generation lock '{}' after {} ms; set {FIXTURE_LOCK_TIMEOUT_ENV} to increase",
                        lock_path.display(),
                        timeout_ms
                    )));
                }
                tokio::time::sleep(Duration::from_millis(retry_ms)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn scale_to_duckdb_factor(scale: &str) -> BenchResult<String> {
    let Some(value) = scale.strip_prefix("sf") else {
        return Err(BenchError::InvalidArgument(format!(
            "unsupported TPC-DS scale '{scale}'; expected format sf<N>"
        )));
    };
    if value.is_empty() || value.parse::<u64>().is_err() {
        return Err(BenchError::InvalidArgument(format!(
            "unsupported TPC-DS scale '{scale}'; expected format sf<N>"
        )));
    }
    Ok(value.to_string())
}

pub async fn generate_fixtures(
    fixtures_dir: &Path,
    scale: &str,
    seed: u64,
    force: bool,
    storage: &StorageConfig,
) -> BenchResult<()> {
    generate_fixtures_with_profile(
        fixtures_dir,
        scale,
        seed,
        force,
        FixtureProfile::Standard,
        storage,
    )
    .await
}

pub async fn generate_fixtures_with_profile(
    fixtures_dir: &Path,
    scale: &str,
    seed: u64,
    force: bool,
    profile: FixtureProfile,
    storage: &StorageConfig,
) -> BenchResult<()> {
    let root = fixture_root(fixtures_dir, scale)?;
    let dataset_dir = root.join("narrow_sales");
    let data_path = dataset_dir.join("rows.jsonl");
    let manifest_path = root.join("manifest.json");
    let rows = scale_to_row_count(scale)?;
    let _scale_lock = acquire_fixture_generation_lock(fixtures_dir, scale).await?;

    if root.exists() && !force {
        if let Ok(existing) = load_manifest(fixtures_dir, scale) {
            let local_tables_ready =
                !storage.is_local() || required_local_fixture_tables_exist(&root);
            let matches_request = existing.schema_version == FIXTURE_SCHEMA_VERSION
                && existing.seed == seed
                && existing.scale == scale
                && existing.rows == rows
                && existing.profile == profile.as_str()
                && local_tables_ready;
            if matches_request {
                return Ok(());
            }
        }
    }
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir_all(&dataset_dir)?;

    let data = generate_narrow_sales_rows(seed, rows);
    write_rows_jsonl(&data_path, &data)?;

    write_delta_table(
        narrow_sales_table_url(fixtures_dir, scale, storage)?,
        &data,
        storage,
    )
    .await?;
    if profile == FixtureProfile::ManyVersions {
        write_many_narrow_sales_versions(
            narrow_sales_table_url(fixtures_dir, scale, storage)?,
            &data,
            storage,
        )
        .await?;
    }

    write_delta_table_partitioned_small_files(
        read_partitioned_table_url(fixtures_dir, scale, storage)?,
        &data,
        128,
        &["region"],
        storage,
    )
    .await?;

    let merge_rows = data
        .iter()
        .take((data.len() / 4).max(1024))
        .cloned()
        .collect::<Vec<_>>();
    write_delta_table(
        merge_target_table_url(fixtures_dir, scale, storage)?,
        &merge_rows,
        storage,
    )
    .await?;

    write_delta_table_partitioned_small_files(
        merge_partitioned_target_table_url(fixtures_dir, scale, storage)?,
        &merge_rows,
        64,
        &["region"],
        storage,
    )
    .await?;

    write_delta_table_partitioned_small_files(
        delete_update_small_files_table_url(fixtures_dir, scale, storage)?,
        &data,
        64,
        &["region"],
        storage,
    )
    .await?;

    let optimize_rows = data
        .iter()
        .take((data.len() / 2).max(2048))
        .cloned()
        .collect::<Vec<_>>();
    write_delta_table_small_files(
        optimize_small_files_table_url(fixtures_dir, scale, storage)?,
        &optimize_rows,
        128,
        storage,
    )
    .await?;

    write_delta_table(
        optimize_compacted_table_url(fixtures_dir, scale, storage)?,
        &optimize_rows,
        storage,
    )
    .await?;

    let vacuum_rows = data
        .iter()
        .take((data.len() / 3).max(1024))
        .cloned()
        .collect::<Vec<_>>();
    write_vacuum_ready_table(
        vacuum_ready_table_url(fixtures_dir, scale, storage)?,
        &vacuum_rows,
        storage,
    )
    .await?;

    let tpcds_store_sales_table_url = tpcds_store_sales_table_url(fixtures_dir, scale, storage)?;
    match profile {
        FixtureProfile::TpcdsDuckdb => {
            write_tpcds_store_sales_table_from_duckdb(tpcds_store_sales_table_url, scale, storage)
                .await?;
        }
        FixtureProfile::Standard | FixtureProfile::ManyVersions => {
            write_tpcds_store_sales_table(tpcds_store_sales_table_url, &data, storage).await?;
        }
    }

    let manifest = FixtureManifest {
        schema_version: FIXTURE_SCHEMA_VERSION,
        seed,
        scale: scale.to_string(),
        rows,
        profile: profile.as_str().to_string(),
    };
    fs::write(manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    Ok(())
}

fn write_rows_jsonl(path: &Path, rows: &[NarrowSaleRow]) -> BenchResult<()> {
    let mut file = fs::File::create(path)?;
    for row in rows {
        let line = serde_json::to_string(row)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

pub(crate) async fn write_delta_table(
    table_url: Url,
    rows: &[NarrowSaleRow],
    storage: &StorageConfig,
) -> BenchResult<()> {
    prepare_local_table_dir(&table_url)?;

    let batch = rows_to_batch(rows)?;
    let _ = storage
        .try_from_url_for_write(table_url)
        .await?
        .write(vec![batch])
        .with_save_mode(SaveMode::Overwrite)
        .await?;

    Ok(())
}

pub(crate) async fn write_delta_table_small_files(
    table_url: Url,
    rows: &[NarrowSaleRow],
    chunk_size: usize,
    storage: &StorageConfig,
) -> BenchResult<()> {
    prepare_local_table_dir(&table_url)?;

    let mut table = storage.try_from_url_for_write(table_url).await?;
    for (idx, chunk) in rows.chunks(chunk_size).enumerate() {
        let mode = if idx == 0 {
            SaveMode::Overwrite
        } else {
            SaveMode::Append
        };
        table = table
            .write(vec![rows_to_batch(chunk)?])
            .with_save_mode(mode)
            .await?;
    }

    Ok(())
}

pub(crate) async fn write_delta_table_partitioned_small_files(
    table_url: Url,
    rows: &[NarrowSaleRow],
    chunk_size: usize,
    partition_columns: &[&str],
    storage: &StorageConfig,
) -> BenchResult<()> {
    prepare_local_table_dir(&table_url)?;

    let mut table = storage.try_from_url_for_write(table_url).await?;
    for (idx, chunk) in rows.chunks(chunk_size).enumerate() {
        let mode = if idx == 0 {
            SaveMode::Overwrite
        } else {
            SaveMode::Append
        };
        table = table
            .write(vec![rows_to_batch(chunk)?])
            .with_save_mode(mode)
            .with_partition_columns(partition_columns.iter().copied())
            .await?;
    }

    Ok(())
}

pub(crate) async fn write_vacuum_ready_table(
    table_url: Url,
    rows: &[NarrowSaleRow],
    storage: &StorageConfig,
) -> BenchResult<()> {
    write_delta_table(table_url.clone(), rows, storage).await?;

    let retained = (rows.len() / 3).max(1);
    let _ = storage
        .try_from_url_for_write(table_url)
        .await?
        .write(vec![rows_to_batch(&rows[..retained])?])
        .with_save_mode(SaveMode::Overwrite)
        .await?;

    Ok(())
}

async fn write_many_narrow_sales_versions(
    table_url: Url,
    rows: &[NarrowSaleRow],
    storage: &StorageConfig,
) -> BenchResult<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let mut table = storage.try_from_url_for_write(table_url).await?;
    let chunk_size = (rows.len() / 64).clamp(32, 256);

    for commit_idx in 0..MANY_VERSIONS_APPEND_COMMITS {
        let start = (commit_idx * chunk_size) % rows.len();
        let end = (start + chunk_size).min(rows.len());
        let mut chunk = rows[start..end].to_vec();
        if chunk.is_empty() {
            chunk.push(rows[commit_idx % rows.len()].clone());
        }
        for row in &mut chunk {
            row.id = row
                .id
                .saturating_add(((commit_idx as i64) + 1) * 1_000_000_000);
            row.ts_ms = row.ts_ms.saturating_add(((commit_idx as i64) + 1) * 60_000);
        }
        table = table
            .write(vec![rows_to_batch(&chunk)?])
            .with_save_mode(SaveMode::Append)
            .await?;
    }

    Ok(())
}

async fn write_tpcds_store_sales_table(
    table_url: Url,
    rows: &[NarrowSaleRow],
    storage: &StorageConfig,
) -> BenchResult<()> {
    prepare_local_table_dir(&table_url)?;

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("ss_customer_sk", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new(
            "ss_ext_sales_price",
            arrow::datatypes::DataType::Float64,
            false,
        ),
        arrow::datatypes::Field::new("ss_item_sk", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("ss_quantity", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("ss_sold_date_sk", arrow::datatypes::DataType::Int64, false),
    ]));

    let ss_customer_sk = rows
        .iter()
        .map(|row| (row.id.rem_euclid(10_000)) + 1)
        .collect::<Vec<_>>();
    let ss_ext_sales_price = rows
        .iter()
        .map(|row| (row.value_i64.abs() as f64 / 10.0) + 1.0)
        .collect::<Vec<_>>();
    let ss_item_sk = rows
        .iter()
        .map(|row| (row.id.rem_euclid(5_000)) + 1)
        .collect::<Vec<_>>();
    let ss_quantity = rows
        .iter()
        .map(|row| row.value_i64.abs().rem_euclid(8) + 1)
        .collect::<Vec<_>>();
    let ss_sold_date_sk = rows
        .iter()
        .map(|row| 2_451_545_i64 + row.id.rem_euclid(3_650))
        .collect::<Vec<_>>();

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow::array::Int64Array::from(ss_customer_sk)),
            Arc::new(arrow::array::Float64Array::from(ss_ext_sales_price)),
            Arc::new(arrow::array::Int64Array::from(ss_item_sk)),
            Arc::new(arrow::array::Int64Array::from(ss_quantity)),
            Arc::new(arrow::array::Int64Array::from(ss_sold_date_sk)),
        ],
    )?;

    let _ = storage
        .try_from_url_for_write(table_url)
        .await?
        .write(vec![batch])
        .with_save_mode(SaveMode::Overwrite)
        .await?;

    Ok(())
}

async fn write_tpcds_store_sales_table_from_duckdb(
    table_url: Url,
    scale: &str,
    storage: &StorageConfig,
) -> BenchResult<()> {
    let runtime = TpcdsDuckdbRuntime::from_env()?;
    let temp_dir = tempfile::tempdir()?;
    let csv_path = temp_dir.path().join("store_sales.csv");

    run_tpcds_duckdb_generator(scale, &runtime, &csv_path).await?;
    write_tpcds_store_sales_csv_table(table_url, csv_path.as_path(), storage).await?;
    Ok(())
}

async fn run_tpcds_duckdb_generator(
    scale: &str,
    runtime: &TpcdsDuckdbRuntime,
    output_csv: &Path,
) -> BenchResult<()> {
    let scale_factor = scale_to_duckdb_factor(scale)?;
    let mut command = tokio::process::Command::new(&runtime.python_executable);
    command.kill_on_drop(true);
    command
        .arg(&runtime.script_path)
        .arg("--scale-factor")
        .arg(&scale_factor)
        .arg("--output-csv")
        .arg(output_csv);

    let output = match tokio::time::timeout(runtime.timeout, command.output()).await {
        Ok(result) => result.map_err(|error| {
            BenchError::InvalidArgument(format!(
                "failed to start duckdb generator using {TPCDS_DUCKDB_PYTHON_ENV}='{}': {error}",
                runtime.python_executable
            ))
        })?,
        Err(_) => {
            return Err(BenchError::InvalidArgument(format!(
                "duckdb generator timed out after {} ms (set {TPCDS_DUCKDB_TIMEOUT_ENV} to increase)",
                runtime.timeout.as_millis()
            )));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(BenchError::InvalidArgument(format!(
            "duckdb generator failed (script='{}', scale_factor={}): {}",
            runtime.script_path.display(),
            scale_factor,
            if stderr.is_empty() {
                "no stderr output".to_string()
            } else {
                stderr
            }
        )));
    }

    if !output_csv.exists() {
        return Err(BenchError::InvalidArgument(format!(
            "duckdb generator did not create CSV output at '{}'",
            output_csv.display()
        )));
    }
    Ok(())
}

async fn write_tpcds_store_sales_csv_table(
    table_url: Url,
    csv_path: &Path,
    storage: &StorageConfig,
) -> BenchResult<()> {
    prepare_local_table_dir(&table_url)?;

    let mut table = storage.try_from_url_for_write(table_url).await?;
    let file = fs::File::open(csv_path)?;
    let mut reader = BufReader::new(file);

    let mut header = String::new();
    if reader.read_line(&mut header)? == 0 {
        return Err(BenchError::InvalidArgument(format!(
            "duckdb generator produced empty CSV at '{}'",
            csv_path.display()
        )));
    }
    let expected_header =
        "ss_customer_sk,ss_ext_sales_price,ss_item_sk,ss_quantity,ss_sold_date_sk";
    if header.trim_end_matches(['\r', '\n']) != expected_header {
        return Err(BenchError::InvalidArgument(format!(
            "duckdb generator CSV header mismatch; expected '{expected_header}'"
        )));
    }

    let mut has_rows = false;
    let mut mode = SaveMode::Overwrite;
    let mut chunk = Vec::with_capacity(TPCDS_DUCKDB_CHUNK_ROWS);
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let row = parse_tpcds_store_sales_row(trimmed, line_idx + 2)?;
        chunk.push(row);
        if chunk.len() >= TPCDS_DUCKDB_CHUNK_ROWS {
            table = table
                .write(vec![tpcds_store_sales_rows_to_batch(&chunk)?])
                .with_save_mode(mode)
                .await?;
            chunk.clear();
            mode = SaveMode::Append;
            has_rows = true;
        }
    }

    if !chunk.is_empty() {
        table = table
            .write(vec![tpcds_store_sales_rows_to_batch(&chunk)?])
            .with_save_mode(mode)
            .await?;
        has_rows = true;
    }

    if !has_rows {
        return Err(BenchError::InvalidArgument(format!(
            "duckdb generator CSV has no data rows: '{}'",
            csv_path.display()
        )));
    }

    // Keep the latest table handle in-scope until writes are durably completed.
    let _ = table;
    Ok(())
}

fn parse_tpcds_store_sales_row(line: &str, line_number: usize) -> BenchResult<TpcdsStoreSalesRow> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 5 {
        return Err(BenchError::InvalidArgument(format!(
            "invalid duckdb generator CSV row at line {line_number}: expected 5 columns, found {}",
            fields.len()
        )));
    }
    Ok(TpcdsStoreSalesRow {
        ss_customer_sk: parse_csv_i64(fields[0], line_number, "ss_customer_sk")?,
        ss_ext_sales_price: parse_csv_f64(fields[1], line_number, "ss_ext_sales_price")?,
        ss_item_sk: parse_csv_i64(fields[2], line_number, "ss_item_sk")?,
        ss_quantity: parse_csv_i64(fields[3], line_number, "ss_quantity")?,
        ss_sold_date_sk: parse_csv_i64(fields[4], line_number, "ss_sold_date_sk")?,
    })
}

fn parse_csv_i64(value: &str, line_number: usize, column: &str) -> BenchResult<i64> {
    value.parse::<i64>().map_err(|error| {
        BenchError::InvalidArgument(format!(
            "invalid duckdb generator CSV value at line {line_number}, column '{column}': {error}"
        ))
    })
}

fn parse_csv_f64(value: &str, line_number: usize, column: &str) -> BenchResult<f64> {
    value.parse::<f64>().map_err(|error| {
        BenchError::InvalidArgument(format!(
            "invalid duckdb generator CSV value at line {line_number}, column '{column}': {error}"
        ))
    })
}

fn tpcds_store_sales_rows_to_batch(
    rows: &[TpcdsStoreSalesRow],
) -> BenchResult<arrow::record_batch::RecordBatch> {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("ss_customer_sk", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new(
            "ss_ext_sales_price",
            arrow::datatypes::DataType::Float64,
            false,
        ),
        arrow::datatypes::Field::new("ss_item_sk", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("ss_quantity", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("ss_sold_date_sk", arrow::datatypes::DataType::Int64, false),
    ]));

    let ss_customer_sk = rows
        .iter()
        .map(|row| row.ss_customer_sk)
        .collect::<Vec<_>>();
    let ss_ext_sales_price = rows
        .iter()
        .map(|row| row.ss_ext_sales_price)
        .collect::<Vec<_>>();
    let ss_item_sk = rows.iter().map(|row| row.ss_item_sk).collect::<Vec<_>>();
    let ss_quantity = rows.iter().map(|row| row.ss_quantity).collect::<Vec<_>>();
    let ss_sold_date_sk = rows
        .iter()
        .map(|row| row.ss_sold_date_sk)
        .collect::<Vec<_>>();

    Ok(arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow::array::Int64Array::from(ss_customer_sk)),
            Arc::new(arrow::array::Float64Array::from(ss_ext_sales_price)),
            Arc::new(arrow::array::Int64Array::from(ss_item_sk)),
            Arc::new(arrow::array::Int64Array::from(ss_quantity)),
            Arc::new(arrow::array::Int64Array::from(ss_sold_date_sk)),
        ],
    )?)
}

fn prepare_local_table_dir(table_url: &Url) -> BenchResult<()> {
    if table_url.scheme() != "file" {
        return Ok(());
    }

    let table_dir = table_url.to_file_path().map_err(|()| {
        BenchError::InvalidArgument(format!("failed to convert URL to file path: {table_url}"))
    })?;

    if table_dir.exists() {
        fs::remove_dir_all(&table_dir)?;
    }
    fs::create_dir_all(&table_dir)?;
    Ok(())
}

pub(crate) fn rows_to_batch(
    rows: &[NarrowSaleRow],
) -> BenchResult<arrow::record_batch::RecordBatch> {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("ts_ms", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("value_i64", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("flag", arrow::datatypes::DataType::Boolean, false),
    ]));

    let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
    let ts_ms: Vec<i64> = rows.iter().map(|r| r.ts_ms).collect();
    let regions: Vec<String> = rows.iter().map(|r| r.region.clone()).collect();
    let values: Vec<i64> = rows.iter().map(|r| r.value_i64).collect();
    let flags: Vec<bool> = rows.iter().map(|r| r.flag).collect();

    Ok(arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow::array::Int64Array::from(ids)),
            Arc::new(arrow::array::Int64Array::from(ts_ms)),
            Arc::new(arrow::array::StringArray::from(regions)),
            Arc::new(arrow::array::Int64Array::from(values)),
            Arc::new(arrow::array::BooleanArray::from(flags)),
        ],
    )?)
}

pub fn load_rows(fixtures_dir: &Path, scale: &str) -> BenchResult<Vec<NarrowSaleRow>> {
    let data_path = fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join("narrow_sales")
        .join("rows.jsonl");

    let data = fs::read_to_string(data_path)?;
    let mut rows = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: NarrowSaleRow = serde_json::from_str(line)?;
        rows.push(row);
    }
    Ok(rows)
}

pub fn load_manifest(fixtures_dir: &Path, scale: &str) -> BenchResult<FixtureManifest> {
    let path = fixture_root(fixtures_dir, scale)
        .expect("validated scale path")
        .join("manifest.json");
    let manifest: FixtureManifest = serde_json::from_slice(&fs::read(path)?)?;
    Ok(manifest)
}
