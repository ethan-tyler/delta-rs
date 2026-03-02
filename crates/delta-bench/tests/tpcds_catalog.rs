use delta_bench::suites::tpcds::catalog::phase1_query_catalog;

#[test]
fn query_ids_are_stable_sorted_and_unique() {
    let specs = phase1_query_catalog();
    let ids = specs.iter().map(|spec| spec.id).collect::<Vec<_>>();

    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();

    assert_eq!(
        ids.len(),
        sorted.len(),
        "query IDs must not contain duplicates"
    );
    assert_eq!(
        ids, sorted,
        "query IDs must be sorted for deterministic output"
    );
}

#[test]
fn q72_is_present_but_disabled_with_explicit_datafusion_issue_reason() {
    let specs = phase1_query_catalog();
    let q72 = specs
        .iter()
        .find(|spec| spec.id == "q72")
        .expect("q72 must remain cataloged explicitly");

    assert!(!q72.enabled, "q72 must be marked disabled");
    let reason = q72
        .skip_reason
        .expect("q72 should include an explicit skip reason");
    let reason_lower = reason.to_ascii_lowercase();
    assert!(
        reason_lower.contains("datafusion") && reason_lower.contains("issue"),
        "q72 skip reason should reference DataFusion issue tracking; got: {reason}"
    );
}

#[test]
fn phase1_catalog_contains_at_least_one_enabled_query() {
    let specs = phase1_query_catalog();
    assert!(
        specs.iter().any(|spec| spec.enabled),
        "phase 1 catalog should contain at least one executable query"
    );
}
