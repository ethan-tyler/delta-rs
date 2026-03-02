use delta_bench::cli::validate_label;

#[test]
fn accepts_safe_label_chars() {
    validate_label("local-main_20260227.1").expect("label should be valid");
}

#[test]
fn rejects_path_traversal_and_separators() {
    for label in ["../escape", "a/b", "a\\b", "", " space"] {
        let err = validate_label(label).expect_err("label should be rejected");
        assert!(
            err.to_string().contains("label"),
            "unexpected error for '{label}': {err}"
        );
    }
}
