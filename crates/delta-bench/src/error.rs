use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("arrow error: {0}")]
    Arrow(#[from] deltalake_core::arrow::error::ArrowError),
    #[error("delta error: {0}")]
    Delta(#[from] deltalake_core::DeltaTableError),
    #[error("datafusion error: {0}")]
    DataFusion(#[from] deltalake_core::datafusion::error::DataFusionError),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type BenchResult<T> = Result<T, BenchError>;
