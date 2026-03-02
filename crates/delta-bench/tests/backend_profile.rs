use std::fs;

use delta_bench::storage::load_backend_profile_options_from_root;

#[test]
fn local_profile_loads_without_env_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(temp.path().join("backends")).expect("create backends dir");

    let options = load_backend_profile_options_from_root(Some("local"), temp.path())
        .expect("local profile should load");
    assert!(options.is_empty());
}

#[test]
fn missing_profile_file_returns_error() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(temp.path().join("backends")).expect("create backends dir");

    let err = load_backend_profile_options_from_root(Some("s3_locking_vultr"), temp.path())
        .expect_err("missing file should error");
    assert!(
        err.to_string().contains("backend profile"),
        "unexpected error: {err}"
    );
}

#[test]
fn profile_file_parses_key_values() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backends = temp.path().join("backends");
    fs::create_dir_all(&backends).expect("create backends dir");
    fs::write(
        backends.join("s3_locking_vultr.env"),
        "# comment\nAWS_REGION=us-east-1\n table_root=s3://bench/private\nDYNAMO_LOCK_TABLE=delta_lock\n",
    )
    .expect("write profile file");

    let options = load_backend_profile_options_from_root(Some("s3_locking_vultr"), temp.path())
        .expect("profile should parse");

    assert_eq!(
        options.get("AWS_REGION").map(String::as_str),
        Some("us-east-1")
    );
    assert_eq!(
        options.get("table_root").map(String::as_str),
        Some("s3://bench/private")
    );
    assert_eq!(
        options.get("DYNAMO_LOCK_TABLE").map(String::as_str),
        Some("delta_lock")
    );
}

#[test]
fn profile_name_rejects_parent_traversal() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::create_dir_all(temp.path().join("backends")).expect("create backends dir");
    fs::write(
        temp.path().join("evil.env"),
        "table_root=s3://unsafe/location\n",
    )
    .expect("write traversal target");

    let err = load_backend_profile_options_from_root(Some("../evil"), temp.path())
        .expect_err("parent traversal profile names must be rejected");
    assert!(
        err.to_string().contains("invalid backend profile"),
        "unexpected error: {err}"
    );
}
