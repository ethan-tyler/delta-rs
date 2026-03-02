use chrono::Utc;
use delta_bench::results::BenchContext;
use delta_bench::system::{benchmark_fidelity_info, FidelityEnvOverrides};
use std::fs;

#[test]
fn bench_context_serializes_optional_fidelity_fields() {
    let ctx = BenchContext {
        schema_version: 2,
        label: "local".to_string(),
        git_sha: Some("deadbeef".to_string()),
        created_at: Utc::now(),
        host: "runner-01".to_string(),
        suite: "all".to_string(),
        scale: "sf1".to_string(),
        iterations: 5,
        warmup: 1,
        dataset_id: Some("tiny_smoke".to_string()),
        dataset_fingerprint: Some("sha256:fixture".to_string()),
        runner: Some("rust".to_string()),
        backend_profile: Some("local".to_string()),
        image_version: Some("image-2026-02-27".to_string()),
        hardening_profile_id: Some("cis-l1-tailored".to_string()),
        hardening_profile_sha256: Some("hardening-sha".to_string()),
        cpu_model: Some("Intel Xeon".to_string()),
        cpu_microcode: Some("0xffffffff".to_string()),
        kernel: Some("6.8.0".to_string()),
        boot_params: Some("quiet".to_string()),
        cpu_steal_pct: Some(0.0),
        numa_topology: Some("nodes=1".to_string()),
        egress_policy_sha256: Some("egress-sha".to_string()),
        run_mode: Some("run-mode".to_string()),
        maintenance_window_id: Some("weekly-sat-0200z".to_string()),
    };

    let raw = serde_json::to_value(ctx).expect("serialize bench context");
    let obj = raw.as_object().expect("json object");

    for key in [
        "image_version",
        "hardening_profile_id",
        "hardening_profile_sha256",
        "cpu_model",
        "cpu_microcode",
        "kernel",
        "boot_params",
        "cpu_steal_pct",
        "numa_topology",
        "egress_policy_sha256",
        "run_mode",
        "maintenance_window_id",
    ] {
        assert!(obj.contains_key(key), "missing key: {key}");
    }
}

#[test]
fn fidelity_info_uses_overrides_for_hash_and_mode_fields() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let hardening = tmp.path().join("tailoring.xml");
    let egress = tmp.path().join("nftables.conf");
    let run_mode = tmp.path().join("security-mode");

    fs::write(&hardening, "<tailoring/>").expect("write hardening file");
    fs::write(&egress, "table inet filter {}").expect("write egress policy file");
    fs::write(&run_mode, "run-mode\n").expect("write run mode file");

    let overrides = FidelityEnvOverrides {
        image_version: Some("img-1".to_string()),
        hardening_profile_id: Some("cis-l1".to_string()),
        hardening_profile_sha256: None,
        hardening_profile_path: Some(hardening),
        egress_policy_sha256: None,
        egress_policy_path: Some(egress),
        run_mode: None,
        run_mode_path: Some(run_mode),
        maintenance_window_id: Some("mw-1".to_string()),
    };

    let info = benchmark_fidelity_info(&overrides);
    assert_eq!(info.image_version.as_deref(), Some("img-1"));
    assert_eq!(info.hardening_profile_id.as_deref(), Some("cis-l1"));
    assert!(info.hardening_profile_sha256.is_some());
    assert!(info.egress_policy_sha256.is_some());
    assert_eq!(info.run_mode.as_deref(), Some("run-mode"));
    assert_eq!(info.maintenance_window_id.as_deref(), Some("mw-1"));
}
