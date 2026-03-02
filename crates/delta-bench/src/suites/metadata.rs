use std::path::Path;

use serde_json::json;
use url::Url;

use super::{copy_dir_all, into_case_result};
use crate::data::fixtures::{narrow_sales_table_path, narrow_sales_table_url};
use crate::error::{BenchError, BenchResult};
use crate::fingerprint::hash_json;
use crate::results::{CaseResult, RuntimeIOMetrics, SampleMetrics};
use crate::runner::{run_case_async, run_case_async_with_setup};
use crate::storage::StorageConfig;

struct MetadataIterationSetup {
    _temp: tempfile::TempDir,
    table_url: Url,
}

pub fn case_names() -> Vec<String> {
    vec![
        "metadata_load".to_string(),
        "metadata_time_travel_v0".to_string(),
    ]
}

pub async fn run(
    fixtures_dir: &Path,
    scale: &str,
    warmup: u32,
    iterations: u32,
    storage: &StorageConfig,
) -> BenchResult<Vec<CaseResult>> {
    if storage.is_local() {
        let table_path = narrow_sales_table_path(fixtures_dir, scale)?;
        let mut out = Vec::new();

        let c1 = run_case_async_with_setup(
            "metadata_load",
            warmup,
            iterations,
            || prepare_metadata_iteration(&table_path).map_err(|e| e.to_string()),
            |setup| {
                let storage = storage.clone();
                async move {
                    let table_url = setup.table_url.clone();
                    let _keep_temp = setup;
                    let table = storage
                        .open_table(table_url)
                        .await
                        .map_err(|e| e.to_string())?;
                    let table_version = table.version().map(|v| v as u64);
                    let result_hash = hash_json(&json!({
                        "operation": "metadata_load",
                        "table_version": table_version,
                    }))
                    .map_err(|e| e.to_string())?;
                    let schema_hash = hash_json(&json!(["operation:string", "table_version:u64",]))
                        .map_err(|e| e.to_string())?;
                    Ok::<SampleMetrics, String>(
                        SampleMetrics::base(None, None, Some(1), table_version).with_runtime_io(
                            RuntimeIOMetrics {
                                peak_rss_mb: None,
                                cpu_time_ms: None,
                                bytes_read: None,
                                bytes_written: None,
                                files_touched: None,
                                files_skipped: None,
                                spill_bytes: None,
                                result_hash: Some(result_hash),
                                schema_hash: Some(schema_hash),
                            },
                        ),
                    )
                }
            },
        )
        .await;
        out.push(into_case_result(c1));

        let c2 = run_case_async_with_setup(
            "metadata_time_travel_v0",
            warmup,
            iterations,
            || prepare_metadata_iteration(&table_path).map_err(|e| e.to_string()),
            |setup| {
                let storage = storage.clone();
                async move {
                    let table_url = setup.table_url.clone();
                    let _keep_temp = setup;
                    let mut table = storage
                        .try_from_url_for_write(table_url)
                        .await
                        .map_err(|e| e.to_string())?;
                    table.load_version(0).await.map_err(|e| e.to_string())?;
                    let table_version = table.version().map(|v| v as u64);
                    let result_hash = hash_json(&json!({
                        "operation": "metadata_time_travel_v0",
                        "table_version": table_version,
                    }))
                    .map_err(|e| e.to_string())?;
                    let schema_hash = hash_json(&json!(["operation:string", "table_version:u64",]))
                        .map_err(|e| e.to_string())?;
                    Ok::<SampleMetrics, String>(
                        SampleMetrics::base(None, None, Some(1), table_version).with_runtime_io(
                            RuntimeIOMetrics {
                                peak_rss_mb: None,
                                cpu_time_ms: None,
                                bytes_read: None,
                                bytes_written: None,
                                files_touched: None,
                                files_skipped: None,
                                spill_bytes: None,
                                result_hash: Some(result_hash),
                                schema_hash: Some(schema_hash),
                            },
                        ),
                    )
                }
            },
        )
        .await;
        out.push(into_case_result(c2));

        return Ok(out);
    }

    let table_url = narrow_sales_table_url(fixtures_dir, scale, storage)?;
    let mut out = Vec::new();

    let c1 = run_case_async("metadata_load", warmup, iterations, || {
        let storage = storage.clone();
        let table_url = table_url.clone();
        async move {
            let table = storage
                .open_table(table_url)
                .await
                .map_err(|e| e.to_string())?;
            let table_version = table.version().map(|v| v as u64);
            let result_hash = hash_json(&json!({
                "operation": "metadata_load",
                "table_version": table_version,
            }))
            .map_err(|e| e.to_string())?;
            let schema_hash = hash_json(&json!(["operation:string", "table_version:u64"]))
                .map_err(|e| e.to_string())?;
            Ok::<SampleMetrics, String>(
                SampleMetrics::base(None, None, Some(1), table_version).with_runtime_io(
                    RuntimeIOMetrics {
                        peak_rss_mb: None,
                        cpu_time_ms: None,
                        bytes_read: None,
                        bytes_written: None,
                        files_touched: None,
                        files_skipped: None,
                        spill_bytes: None,
                        result_hash: Some(result_hash),
                        schema_hash: Some(schema_hash),
                    },
                ),
            )
        }
    })
    .await;
    out.push(into_case_result(c1));

    let c2 = run_case_async("metadata_time_travel_v0", warmup, iterations, || {
        let storage = storage.clone();
        let table_url = table_url.clone();
        async move {
            let mut table = storage
                .try_from_url_for_write(table_url)
                .await
                .map_err(|e| e.to_string())?;
            table.load_version(0).await.map_err(|e| e.to_string())?;
            let table_version = table.version().map(|v| v as u64);
            let result_hash = hash_json(&json!({
                "operation": "metadata_time_travel_v0",
                "table_version": table_version,
            }))
            .map_err(|e| e.to_string())?;
            let schema_hash = hash_json(&json!(["operation:string", "table_version:u64"]))
                .map_err(|e| e.to_string())?;
            Ok::<SampleMetrics, String>(
                SampleMetrics::base(None, None, Some(1), table_version).with_runtime_io(
                    RuntimeIOMetrics {
                        peak_rss_mb: None,
                        cpu_time_ms: None,
                        bytes_read: None,
                        bytes_written: None,
                        files_touched: None,
                        files_skipped: None,
                        spill_bytes: None,
                        result_hash: Some(result_hash),
                        schema_hash: Some(schema_hash),
                    },
                ),
            )
        }
    })
    .await;
    out.push(into_case_result(c2));

    Ok(out)
}

fn prepare_metadata_iteration(source_table_path: &Path) -> BenchResult<MetadataIterationSetup> {
    let temp = tempfile::tempdir()?;
    let table_dir = temp.path().join("table");
    copy_dir_all(source_table_path, &table_dir)?;
    let table_url = Url::from_directory_path(&table_dir).map_err(|()| {
        BenchError::InvalidArgument(format!(
            "failed to create table URL for {}",
            table_dir.display()
        ))
    })?;
    Ok(MetadataIterationSetup {
        _temp: temp,
        table_url,
    })
}
