use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn host_name() -> String {
    if let Ok(output) = Command::new("uname").arg("-n").output() {
        let v = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    "unknown-host".to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FidelityEnvOverrides {
    pub image_version: Option<String>,
    pub hardening_profile_id: Option<String>,
    pub hardening_profile_sha256: Option<String>,
    pub hardening_profile_path: Option<PathBuf>,
    pub egress_policy_sha256: Option<String>,
    pub egress_policy_path: Option<PathBuf>,
    pub run_mode: Option<String>,
    pub run_mode_path: Option<PathBuf>,
    pub maintenance_window_id: Option<String>,
}

impl FidelityEnvOverrides {
    pub fn from_env() -> Self {
        Self {
            image_version: std::env::var("DELTA_BENCH_IMAGE_VERSION").ok(),
            hardening_profile_id: std::env::var("DELTA_BENCH_HARDENING_PROFILE_ID").ok(),
            hardening_profile_sha256: std::env::var("DELTA_BENCH_HARDENING_PROFILE_SHA256").ok(),
            hardening_profile_path: std::env::var("DELTA_BENCH_HARDENING_PROFILE_PATH")
                .ok()
                .map(PathBuf::from),
            egress_policy_sha256: std::env::var("DELTA_BENCH_EGRESS_POLICY_SHA256").ok(),
            egress_policy_path: std::env::var("DELTA_BENCH_EGRESS_POLICY_PATH")
                .ok()
                .map(PathBuf::from),
            run_mode: std::env::var("DELTA_BENCH_RUN_MODE").ok(),
            run_mode_path: std::env::var("DELTA_BENCH_RUN_MODE_PATH")
                .ok()
                .map(PathBuf::from),
            maintenance_window_id: std::env::var("DELTA_BENCH_MAINTENANCE_WINDOW_ID").ok(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BenchmarkFidelityInfo {
    pub image_version: Option<String>,
    pub hardening_profile_id: Option<String>,
    pub hardening_profile_sha256: Option<String>,
    pub cpu_model: Option<String>,
    pub cpu_microcode: Option<String>,
    pub kernel: Option<String>,
    pub boot_params: Option<String>,
    pub cpu_steal_pct: Option<f64>,
    pub numa_topology: Option<String>,
    pub egress_policy_sha256: Option<String>,
    pub run_mode: Option<String>,
    pub maintenance_window_id: Option<String>,
}

pub fn benchmark_fidelity_info(overrides: &FidelityEnvOverrides) -> BenchmarkFidelityInfo {
    let default_hardening_path = PathBuf::from("/etc/delta-bench/cis-tailoring.xml");
    let default_egress_path = PathBuf::from("/etc/nftables.conf");
    let default_run_mode_path = PathBuf::from("/etc/delta-bench/security-mode");
    let default_image_version_path = PathBuf::from("/etc/delta-bench/image-version");
    let default_hardening_id_path = PathBuf::from("/etc/delta-bench/hardening-profile-id");

    let hardening_profile_path = overrides
        .hardening_profile_path
        .clone()
        .unwrap_or(default_hardening_path);
    let egress_policy_path = overrides
        .egress_policy_path
        .clone()
        .unwrap_or(default_egress_path);
    let run_mode_path = overrides
        .run_mode_path
        .clone()
        .unwrap_or(default_run_mode_path);

    BenchmarkFidelityInfo {
        image_version: overrides
            .image_version
            .clone()
            .or_else(|| read_trimmed_file(&default_image_version_path)),
        hardening_profile_id: overrides
            .hardening_profile_id
            .clone()
            .or_else(|| read_trimmed_file(&default_hardening_id_path)),
        hardening_profile_sha256: overrides
            .hardening_profile_sha256
            .clone()
            .or_else(|| sha256_file(&hardening_profile_path)),
        cpu_model: cpu_info_field("model name")
            .or_else(|| sysctl_value("machdep.cpu.brand_string")),
        cpu_microcode: cpu_info_field("microcode"),
        kernel: kernel_release(),
        boot_params: read_trimmed_file(&PathBuf::from("/proc/cmdline")),
        cpu_steal_pct: cpu_steal_percent(),
        numa_topology: numa_topology_summary(),
        egress_policy_sha256: overrides
            .egress_policy_sha256
            .clone()
            .or_else(|| sha256_file(&egress_policy_path)),
        run_mode: overrides
            .run_mode
            .clone()
            .or_else(|| read_trimmed_file(&run_mode_path)),
        maintenance_window_id: overrides.maintenance_window_id.clone(),
    }
}

fn read_trimmed_file(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let value = raw.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn sha256_file(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let digest = Sha256::digest(bytes);
    Some(format!("{digest:x}"))
}

fn cpu_info_field(field_name: &str) -> Option<String> {
    let content = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        let Some((lhs, rhs)) = line.split_once(':') else {
            continue;
        };
        if lhs.trim() == field_name {
            let value = rhs.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn sysctl_value(name: &str) -> Option<String> {
    let output = Command::new("sysctl").arg("-n").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn kernel_release() -> Option<String> {
    let output = Command::new("uname").arg("-r").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn cpu_steal_percent() -> Option<f64> {
    let content = fs::read_to_string("/proc/stat").ok()?;
    let cpu_line = content.lines().find(|line| line.starts_with("cpu "))?;
    let mut fields = cpu_line.split_whitespace();
    let _cpu = fields.next()?;
    let numbers: Vec<f64> = fields.filter_map(|v| v.parse::<f64>().ok()).collect();
    if numbers.len() < 8 {
        return None;
    }
    let total: f64 = numbers.iter().sum();
    if total <= 0.0 {
        return None;
    }
    Some((numbers[7] / total) * 100.0)
}

fn numa_topology_summary() -> Option<String> {
    if let Ok(output) = Command::new("lscpu").output() {
        if output.status.success() {
            let parsed = parse_lscpu_numa(&String::from_utf8_lossy(&output.stdout));
            if parsed.is_some() {
                return parsed;
            }
        }
    }

    let node_dir = PathBuf::from("/sys/devices/system/node");
    let entries = fs::read_dir(node_dir).ok()?;
    let count = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("node"))
        .count();
    if count > 0 {
        Some(format!("nodes={count}"))
    } else {
        None
    }
}

fn parse_lscpu_numa(content: &str) -> Option<String> {
    let mut nodes: Option<String> = None;
    let mut details: Vec<String> = Vec::new();

    for line in content.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key == "NUMA node(s)" {
            nodes = Some(value.to_string());
            continue;
        }
        if key.starts_with("NUMA node") && key.ends_with("CPU(s)") {
            details.push(format!("{key}={value}"));
        }
    }

    let nodes = nodes?;
    if details.is_empty() {
        Some(format!("nodes={nodes}"))
    } else {
        Some(format!("nodes={nodes}; {}", details.join("; ")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaRsCheckoutInfo {
    pub checkout_dir: PathBuf,
    pub checkout_present: bool,
    pub core_present: bool,
}

pub fn delta_rs_checkout_info(path_override: Option<&Path>) -> DeltaRsCheckoutInfo {
    let checkout_dir = match path_override {
        Some(path) => path.to_path_buf(),
        None => std::env::var("DELTA_RS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".delta-rs-under-test")),
    };

    let checkout_present = checkout_dir.exists();
    let core_present = checkout_dir.join("crates/core").exists();

    DeltaRsCheckoutInfo {
        checkout_dir,
        checkout_present,
        core_present,
    }
}
