use std::path::Path;
use std::sync::Arc;

use deltalake_core::protocol::SaveMode;
use deltalake_core::DeltaTable;
use serde_json::json;
use url::Url;

use super::{fixture_error_cases, into_case_result};
use crate::data::fixtures::{load_rows, rows_to_batch};
use crate::error::{BenchError, BenchResult};
use crate::fingerprint::hash_json;
use crate::results::{CaseResult, RuntimeIOMetrics, SampleMetrics};
use crate::runner::run_case_async;
use crate::storage::StorageConfig;

pub fn case_names() -> Vec<String> {
    vec![
        "write_append_small".to_string(),
        "write_append_large".to_string(),
        "write_overwrite".to_string(),
    ]
}

pub async fn run(
    fixtures_dir: &Path,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    if !storage.is_local() {
        return Ok(fixture_error_cases(
            case_names(),
            "write suite does not support non-local storage backend yet",
        ));
    }

    let rows = match load_rows(fixtures_dir, scale) {
        Ok(rows) => Arc::new(rows),
        Err(e) => return Ok(fixture_error_cases(case_names(), &e.to_string())),
    };
    let mut results = Vec::new();

    let small = run_case_async("write_append_small", warmup, iterations, || {
        let rows = Arc::clone(&rows);
        async move {
            run_append_case(rows.as_slice(), 128)
                .await
                .map_err(|e| e.to_string())
        }
    })
    .await;
    results.push(into_case_result(small));

    let large = run_case_async("write_append_large", warmup, iterations, || {
        let rows = Arc::clone(&rows);
        async move {
            run_append_case(rows.as_slice(), 4096)
                .await
                .map_err(|e| e.to_string())
        }
    })
    .await;
    results.push(into_case_result(large));

    let overwrite = run_case_async("write_overwrite", warmup, iterations, || {
        let rows = Arc::clone(&rows);
        async move {
            run_overwrite_case(rows.as_slice())
                .await
                .map_err(|e| e.to_string())
        }
    })
    .await;
    results.push(into_case_result(overwrite));

    Ok(results)
}

async fn run_append_case(
    rows: &[crate::data::datasets::NarrowSaleRow],
    chunk: usize,
) -> BenchResult<SampleMetrics> {
    let temp = tempfile::tempdir()?;
    let table_url = Url::from_directory_path(temp.path()).map_err(|()| {
        BenchError::InvalidArgument(format!(
            "failed to create URL for {}",
            temp.path().display()
        ))
    })?;

    let mut operations = 0_u64;
    let mut table = DeltaTable::try_from_url(table_url).await?;
    for (idx, r) in rows.chunks(chunk).enumerate() {
        operations += 1;
        let mode = if idx == 0 {
            SaveMode::Overwrite
        } else {
            SaveMode::Append
        };
        let batch = rows_to_batch(r)?;
        table = table.write(vec![batch]).with_save_mode(mode).await?;
    }

    let table_version = table.version().map(|v| v as u64);
    let result_hash = hash_json(&json!({
        "rows_processed": rows.len() as u64,
        "operations": operations,
        "table_version": table_version,
    }))?;
    let schema_hash = hash_json(&json!([
        "rows_processed:u64",
        "operations:u64",
        "table_version:u64",
    ]))?;

    Ok(SampleMetrics::base(
        Some(rows.len() as u64),
        None,
        Some(operations),
        table_version,
    )
    .with_runtime_io(RuntimeIOMetrics {
        peak_rss_mb: None,
        cpu_time_ms: None,
        bytes_read: None,
        bytes_written: None,
        files_touched: None,
        files_skipped: None,
        spill_bytes: None,
        result_hash: Some(result_hash),
        schema_hash: Some(schema_hash),
    }))
}

async fn run_overwrite_case(
    rows: &[crate::data::datasets::NarrowSaleRow],
) -> BenchResult<SampleMetrics> {
    let temp = tempfile::tempdir()?;
    let table_url = Url::from_directory_path(temp.path()).map_err(|()| {
        BenchError::InvalidArgument(format!(
            "failed to create URL for {}",
            temp.path().display()
        ))
    })?;
    let mut table = DeltaTable::try_from_url(table_url).await?;

    let first = rows_to_batch(rows)?;
    table = table
        .write(vec![first])
        .with_save_mode(SaveMode::Overwrite)
        .await?;

    let next = rows_to_batch(rows)?;
    table = table
        .write(vec![next])
        .with_save_mode(SaveMode::Overwrite)
        .await?;

    let table_version = table.version().map(|v| v as u64);
    let result_hash = hash_json(&json!({
        "rows_processed": (rows.len() as u64) * 2,
        "operations": 2_u64,
        "table_version": table_version,
    }))?;
    let schema_hash = hash_json(&json!([
        "rows_processed:u64",
        "operations:u64",
        "table_version:u64",
    ]))?;

    Ok(
        SampleMetrics::base(Some((rows.len() as u64) * 2), None, Some(2), table_version)
            .with_runtime_io(RuntimeIOMetrics {
                peak_rss_mb: None,
                cpu_time_ms: None,
                bytes_read: None,
                bytes_written: None,
                files_touched: None,
                files_skipped: None,
                spill_bytes: None,
                result_hash: Some(result_hash),
                schema_hash: Some(schema_hash),
            }),
    )
}
