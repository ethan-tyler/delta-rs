use std::path::Path;
use std::sync::Arc;

use delta_bench::data::fixtures::generate_fixtures;
use delta_bench::storage::StorageConfig;
use delta_bench::suites::tpcds;
use deltalake_core::arrow::array::{Float64Array, Int64Array};
use deltalake_core::arrow::datatypes::{DataType, Field, Schema};
use deltalake_core::arrow::record_batch::RecordBatch;
use deltalake_core::protocol::SaveMode;
use deltalake_core::DeltaTable;
use url::Url;

#[tokio::test]
async fn enabled_queries_execute_and_emit_successful_cases() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_store_sales_fixture(temp.path(), "sf1").await;
    let storage = StorageConfig::local();

    let cases = tpcds::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run tpcds");

    let enabled = cases
        .iter()
        .filter(|case| case.case != "tpcds_q72")
        .collect::<Vec<_>>();
    assert!(!enabled.is_empty(), "expected enabled TPC-DS query cases");
    assert!(
        enabled.iter().all(|case| case.success),
        "enabled queries should succeed: {enabled:#?}"
    );
}

#[tokio::test]
async fn generated_fixtures_provide_required_tpcds_tables() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = StorageConfig::local();
    generate_fixtures(temp.path(), "sf1", 42, true, &storage)
        .await
        .expect("generate fixtures");

    let cases = tpcds::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run tpcds");
    let enabled = cases
        .iter()
        .filter(|case| case.case != "tpcds_q72")
        .collect::<Vec<_>>();
    assert!(
        enabled.iter().all(|case| case.success),
        "generated fixtures should satisfy enabled TPC-DS cases: {enabled:#?}"
    );
}

#[tokio::test]
async fn q72_is_reported_as_deterministically_skipped() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_store_sales_fixture(temp.path(), "sf1").await;
    let storage = StorageConfig::local();

    let cases = tpcds::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run tpcds");
    let q72 = cases
        .iter()
        .find(|case| case.case == "tpcds_q72")
        .expect("q72 result should exist");
    assert!(!q72.success, "q72 should be emitted as skipped/failure");
    let failure = q72
        .failure
        .as_ref()
        .expect("q72 failure payload should be present");
    assert!(
        failure.message.to_ascii_lowercase().contains("skipped"),
        "q72 failure message should encode skip behavior; got: {}",
        failure.message
    );
}

#[tokio::test]
async fn successful_samples_include_normalized_and_scan_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_store_sales_fixture(temp.path(), "sf1").await;
    let storage = StorageConfig::local();

    let cases = tpcds::run(temp.path(), "sf1", 0, 1, &storage)
        .await
        .expect("run tpcds");

    let samples = cases
        .iter()
        .filter(|case| case.success)
        .flat_map(|case| case.samples.iter())
        .collect::<Vec<_>>();
    assert!(
        !samples.is_empty(),
        "expected at least one successful sample"
    );

    for sample in samples {
        let metrics = sample
            .metrics
            .as_ref()
            .expect("successful samples should include metrics");
        assert!(
            metrics.rows_processed.is_some(),
            "rows_processed should be normalized in metrics"
        );
        assert!(
            metrics.files_scanned.is_some()
                || metrics.files_pruned.is_some()
                || metrics.bytes_scanned.is_some()
                || metrics.scan_time_ms.is_some(),
            "expected at least one scan metric in sample: {metrics:?}"
        );
    }
}

async fn write_store_sales_fixture(fixtures_dir: &Path, scale: &str) {
    let table_dir = fixtures_dir.join(scale).join("tpcds").join("store_sales");
    std::fs::create_dir_all(&table_dir).expect("create fixture table dir");

    let table_url = Url::from_directory_path(&table_dir).expect("table url");
    let schema = Arc::new(Schema::new(vec![
        Field::new("ss_customer_sk", DataType::Int64, false),
        Field::new("ss_ext_sales_price", DataType::Float64, false),
        Field::new("ss_item_sk", DataType::Int64, false),
        Field::new("ss_quantity", DataType::Int64, false),
        Field::new("ss_sold_date_sk", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 15.0])),
            Arc::new(Int64Array::from(vec![100, 101, 100, 102])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![2450815, 2450816, 2450817, 2450818])),
        ],
    )
    .expect("record batch");

    let _ = DeltaTable::try_from_url(table_url)
        .await
        .expect("open table")
        .write(vec![batch])
        .with_save_mode(SaveMode::Overwrite)
        .await
        .expect("write fixture");
}
