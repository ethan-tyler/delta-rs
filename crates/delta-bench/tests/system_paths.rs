use std::fs;

use delta_bench::system::delta_rs_checkout_info;

#[test]
fn delta_rs_checkout_info_uses_env_override() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("under-test");
    fs::create_dir_all(root.join("crates/core")).expect("mkdir core");

    let info = delta_rs_checkout_info(Some(root.as_path()));
    assert!(info.checkout_present);
    assert!(info.core_present);
}
