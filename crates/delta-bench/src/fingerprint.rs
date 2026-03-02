use deltalake_core::arrow::datatypes::Schema;
use deltalake_core::arrow::record_batch::RecordBatch;
use deltalake_core::arrow::util::display::array_value_to_string;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::BenchResult;

pub fn hash_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

pub fn hash_display(value: impl std::fmt::Display) -> String {
    hash_bytes(value.to_string().as_bytes())
}

pub fn hash_json<T: Serialize>(value: &T) -> BenchResult<String> {
    let encoded = serde_json::to_vec(value)?;
    Ok(hash_bytes(&encoded))
}

pub fn hash_record_batches_unordered(batches: &[RecordBatch]) -> BenchResult<String> {
    let mut rows = Vec::<String>::new();
    for batch in batches {
        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let value = array_value_to_string(col.as_ref(), row_idx)?;
                row.push((field.name().clone(), value));
            }
            rows.push(serde_json::to_string(&row)?);
        }
    }
    rows.sort();
    hash_json(&rows)
}

pub fn hash_arrow_schema(schema: &Schema) -> BenchResult<String> {
    let schema_rows = schema
        .fields()
        .iter()
        .map(|field| {
            (
                field.name().to_string(),
                field.data_type().to_string(),
                field.is_nullable(),
            )
        })
        .collect::<Vec<_>>();
    hash_json(&schema_rows)
}

pub fn hash_record_batch_schema(batches: &[RecordBatch]) -> BenchResult<String> {
    let Some(schema) = batches.first().map(|batch| batch.schema()) else {
        return hash_json(&Vec::<String>::new());
    };
    hash_arrow_schema(schema.as_ref())
}
