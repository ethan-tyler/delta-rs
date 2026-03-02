use delta_bench::suites::merge::{merge_case_by_name, merge_case_names};

#[test]
fn known_merge_case_names_are_present() {
    let names = merge_case_names();
    assert!(names
        .iter()
        .any(|name| name.contains("merge_upsert_10pct_insert_10pct")));
    assert!(names.iter().any(|name| name.contains("merge_delete_5pct")));
}

#[test]
fn merge_case_lookup_is_case_insensitive() {
    let c = merge_case_by_name("MERGE_DELETE_5PCT");
    assert!(c.is_some());
}
