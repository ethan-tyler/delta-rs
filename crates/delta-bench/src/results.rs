use chrono::{DateTime, Utc};
use serde::{de, Deserialize, Deserializer, Serialize};

fn deserialize_schema_version_v2<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let value = u32::deserialize(deserializer)?;
    if value == 2 {
        Ok(value)
    } else {
        Err(de::Error::custom(format!(
            "schema_version must be 2 (found {value})"
        )))
    }
}

fn deserialize_case_classification<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    parse_case_classification(&value).map_err(de::Error::custom)
}

fn parse_case_classification(value: &str) -> Result<String, String> {
    match value {
        "supported" | "expected_failure" => Ok(value.to_string()),
        other => Err(format!(
            "classification must be one of: supported, expected_failure (found {other})"
        )),
    }
}

pub fn validate_case_classification(value: &str) -> Result<(), String> {
    parse_case_classification(value).map(|_| ())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchContext {
    #[serde(deserialize_with = "deserialize_schema_version_v2")]
    pub schema_version: u32,
    pub label: String,
    pub git_sha: Option<String>,
    pub created_at: DateTime<Utc>,
    pub host: String,
    pub suite: String,
    pub scale: String,
    pub iterations: u32,
    pub warmup: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardening_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardening_profile_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_microcode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boot_params: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_steal_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_topology: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub egress_policy_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maintenance_window_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleMetrics {
    pub rows_processed: Option<u64>,
    pub bytes_processed: Option<u64>,
    pub operations: Option<u64>,
    pub table_version: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_scanned: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_pruned: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_scanned: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_time_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rewrite_time_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_time_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_read: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_written: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_touched: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_skipped: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spill_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_hash: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanRewriteMetrics {
    pub files_scanned: Option<u64>,
    pub files_pruned: Option<u64>,
    pub bytes_scanned: Option<u64>,
    pub scan_time_ms: Option<u64>,
    pub rewrite_time_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIOMetrics {
    pub peak_rss_mb: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub bytes_read: Option<u64>,
    pub bytes_written: Option<u64>,
    pub files_touched: Option<u64>,
    pub files_skipped: Option<u64>,
    pub spill_bytes: Option<u64>,
    pub result_hash: Option<String>,
    pub schema_hash: Option<String>,
}

impl SampleMetrics {
    pub fn base(
        rows_processed: Option<u64>,
        bytes_processed: Option<u64>,
        operations: Option<u64>,
        table_version: Option<u64>,
    ) -> Self {
        Self {
            rows_processed,
            bytes_processed,
            operations,
            table_version,
            files_scanned: None,
            files_pruned: None,
            bytes_scanned: None,
            scan_time_ms: None,
            rewrite_time_ms: None,
            peak_rss_mb: None,
            cpu_time_ms: None,
            bytes_read: None,
            bytes_written: None,
            files_touched: None,
            files_skipped: None,
            spill_bytes: None,
            result_hash: None,
            schema_hash: None,
        }
    }

    pub fn with_scan_rewrite(mut self, metrics: ScanRewriteMetrics) -> Self {
        self.files_scanned = metrics.files_scanned;
        self.files_pruned = metrics.files_pruned;
        self.bytes_scanned = metrics.bytes_scanned;
        self.scan_time_ms = metrics.scan_time_ms;
        self.rewrite_time_ms = metrics.rewrite_time_ms;
        self
    }

    pub fn with_scan_rewrite_metrics(
        self,
        files_scanned: Option<u64>,
        files_pruned: Option<u64>,
        bytes_scanned: Option<u64>,
        scan_time_ms: Option<u64>,
        rewrite_time_ms: Option<u64>,
    ) -> Self {
        self.with_scan_rewrite(ScanRewriteMetrics {
            files_scanned,
            files_pruned,
            bytes_scanned,
            scan_time_ms,
            rewrite_time_ms,
        })
    }

    pub fn with_runtime_io(mut self, metrics: RuntimeIOMetrics) -> Self {
        self.peak_rss_mb = metrics.peak_rss_mb;
        self.cpu_time_ms = metrics.cpu_time_ms;
        self.bytes_read = metrics.bytes_read;
        self.bytes_written = metrics.bytes_written;
        self.files_touched = metrics.files_touched;
        self.files_skipped = metrics.files_skipped;
        self.spill_bytes = metrics.spill_bytes;
        self.result_hash = metrics.result_hash;
        self.schema_hash = metrics.schema_hash;
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_runtime_io_metrics(
        self,
        peak_rss_mb: Option<u64>,
        cpu_time_ms: Option<u64>,
        bytes_read: Option<u64>,
        bytes_written: Option<u64>,
        files_touched: Option<u64>,
        files_skipped: Option<u64>,
        spill_bytes: Option<u64>,
        result_hash: Option<String>,
        schema_hash: Option<String>,
    ) -> Self {
        self.with_runtime_io(RuntimeIOMetrics {
            peak_rss_mb,
            cpu_time_ms,
            bytes_read,
            bytes_written,
            files_touched,
            files_skipped,
            spill_bytes,
            result_hash,
            schema_hash,
        })
    }
}

impl From<u64> for SampleMetrics {
    fn from(rows: u64) -> Self {
        Self::base(Some(rows), None, None, None)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationSample {
    pub elapsed_ms: f64,
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<SampleMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseFailure {
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElapsedStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub stddev_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cv_pct: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseResult {
    pub case: String,
    pub success: bool,
    #[serde(deserialize_with = "deserialize_case_classification")]
    pub classification: String,
    pub samples: Vec<IterationSample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_stats: Option<ElapsedStats>,
    pub failure: Option<CaseFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchRunResult {
    #[serde(deserialize_with = "deserialize_schema_version_v2")]
    pub schema_version: u32,
    pub context: BenchContext,
    pub cases: Vec<CaseResult>,
}
