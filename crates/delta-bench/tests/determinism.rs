use delta_bench::data::generator::generate_narrow_sales_rows;

#[test]
fn same_seed_produces_same_rows() {
    let a = generate_narrow_sales_rows(42, 16);
    let b = generate_narrow_sales_rows(42, 16);
    assert_eq!(a, b);
}

#[test]
fn different_seed_produces_different_rows() {
    let a = generate_narrow_sales_rows(42, 16);
    let b = generate_narrow_sales_rows(43, 16);
    assert_ne!(a, b);
}
