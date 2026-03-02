use delta_bench::fingerprint::hash_arrow_schema;
use deltalake_core::arrow::datatypes::{DataType, Field, Schema};

#[test]
fn hash_arrow_schema_is_deterministic() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, true),
    ]);

    let first = hash_arrow_schema(&schema).expect("hash should succeed");
    let second = hash_arrow_schema(&schema).expect("hash should succeed");
    assert_eq!(first, second);
}

#[test]
fn hash_arrow_schema_changes_when_schema_changes() {
    let left = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, true),
    ]);
    let right = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::LargeUtf8, true),
    ]);

    let left_hash = hash_arrow_schema(&left).expect("hash should succeed");
    let right_hash = hash_arrow_schema(&right).expect("hash should succeed");
    assert_ne!(left_hash, right_hash);
}
