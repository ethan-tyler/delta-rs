use std::sync::Arc;

use deltalake_core::datafusion::physical_plan::metrics::{MetricValue, MetricsSet};
use deltalake_core::datafusion::physical_plan::ExecutionPlan;

#[derive(Default)]
pub(crate) struct ScanMetrics {
    pub(crate) files_scanned: Option<u64>,
    pub(crate) files_pruned: Option<u64>,
    pub(crate) bytes_scanned: Option<u64>,
    pub(crate) scan_time_ms: Option<u64>,
}

pub(crate) fn extract_scan_metrics(plan: &Arc<dyn ExecutionPlan>) -> ScanMetrics {
    let mut files_scanned_total = 0_u64;
    let mut files_scanned_seen = false;
    let mut files_pruned_total = 0_u64;
    let mut files_pruned_seen = false;
    let mut bytes_scanned_total = 0_u64;
    let mut bytes_scanned_seen = false;
    let mut scan_elapsed_nanos_total = 0_u64;
    let mut scan_elapsed_seen = false;

    collect_scan_metrics(
        plan,
        &mut files_scanned_total,
        &mut files_scanned_seen,
        &mut files_pruned_total,
        &mut files_pruned_seen,
        &mut bytes_scanned_total,
        &mut bytes_scanned_seen,
        &mut scan_elapsed_nanos_total,
        &mut scan_elapsed_seen,
    );

    ScanMetrics {
        files_scanned: files_scanned_seen.then_some(files_scanned_total),
        files_pruned: files_pruned_seen.then_some(files_pruned_total),
        bytes_scanned: bytes_scanned_seen.then_some(bytes_scanned_total),
        scan_time_ms: scan_elapsed_seen.then_some(scan_elapsed_nanos_total / 1_000_000),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_scan_metrics(
    plan: &Arc<dyn ExecutionPlan>,
    files_scanned_total: &mut u64,
    files_scanned_seen: &mut bool,
    files_pruned_total: &mut u64,
    files_pruned_seen: &mut bool,
    bytes_scanned_total: &mut u64,
    bytes_scanned_seen: &mut bool,
    scan_elapsed_nanos_total: &mut u64,
    scan_elapsed_seen: &mut bool,
) {
    if let Some(metrics) = plan.metrics() {
        if let Some(v) = sum_count_metrics(&metrics, &["files_scanned", "count_files_scanned"]) {
            *files_scanned_total = files_scanned_total.saturating_add(v);
            *files_scanned_seen = true;
        }
        if let Some(v) = sum_count_metrics(&metrics, &["files_pruned", "count_files_pruned"]) {
            *files_pruned_total = files_pruned_total.saturating_add(v);
            *files_pruned_seen = true;
        }
        if let Some(v) = sum_pruned_metrics(&metrics, &["files_ranges_pruned_statistics"]) {
            *files_pruned_total = files_pruned_total.saturating_add(v);
            *files_pruned_seen = true;
        }
        if let Some(v) = sum_count_metrics(&metrics, &["bytes_scanned"]) {
            *bytes_scanned_total = bytes_scanned_total.saturating_add(v);
            *bytes_scanned_seen = true;
        }

        let is_scan_node = has_metric_name(&metrics, &["files_scanned", "count_files_scanned"])
            || has_metric_name(&metrics, &["bytes_scanned"])
            || has_metric_name(&metrics, &["files_ranges_pruned_statistics"]);
        if is_scan_node {
            if let Some(elapsed_nanos) = metrics.elapsed_compute() {
                *scan_elapsed_nanos_total =
                    scan_elapsed_nanos_total.saturating_add(elapsed_nanos as u64);
                *scan_elapsed_seen = true;
            }
        }
    }

    for child in plan.children() {
        collect_scan_metrics(
            child,
            files_scanned_total,
            files_scanned_seen,
            files_pruned_total,
            files_pruned_seen,
            bytes_scanned_total,
            bytes_scanned_seen,
            scan_elapsed_nanos_total,
            scan_elapsed_seen,
        );
    }
}

fn has_metric_name(metrics: &MetricsSet, names: &[&str]) -> bool {
    metrics.iter().any(|metric| {
        let name = metric.value().name();
        names.contains(&name)
    })
}

fn sum_count_metrics(metrics: &MetricsSet, names: &[&str]) -> Option<u64> {
    let mut total = 0_u64;
    let mut seen = false;
    for metric in metrics.iter() {
        if let MetricValue::Count { name, count } = metric.value() {
            if names.iter().any(|candidate| *candidate == name.as_ref()) {
                total = total.saturating_add(count.value() as u64);
                seen = true;
            }
        }
    }
    seen.then_some(total)
}

pub(crate) fn sum_pruned_metrics(metrics: &MetricsSet, names: &[&str]) -> Option<u64> {
    let mut total = 0_u64;
    let mut seen = false;
    for metric in metrics.iter() {
        if let MetricValue::PruningMetrics {
            name,
            pruning_metrics,
        } = metric.value()
        {
            if names.iter().any(|candidate| *candidate == name.as_ref()) {
                total = total.saturating_add(pruning_metrics.pruned() as u64);
                seen = true;
            }
        }
    }
    seen.then_some(total)
}
