use std::sync::Arc;

use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use deltalake::arrow::array::{
    ArrayRef, Int32Builder, Int64Builder, RecordBatchReader, StringBuilder,
};
use deltalake::arrow::record_batch::RecordBatch;
use deltalake::kernel::{DeletionVectorDescriptor, LogicalFileView, StorageType};
use deltalake::logstore::LogStoreRef;
use object_store::path::Path as ObjectStorePath;
use url::Url;

fn table_root_url(log_store: &LogStoreRef) -> Url {
    let mut root = log_store.root_url().clone();
    if !root.path().ends_with('/') {
        root.set_path(&format!("{}/", root.path()));
    }
    root
}

fn object_store_path(decoded_path: &str) -> ObjectStorePath {
    match ObjectStorePath::parse(decoded_path) {
        Ok(path) => path,
        Err(_) => ObjectStorePath::from(decoded_path),
    }
}

fn file_uri(log_store: &LogStoreRef, decoded_path: &str) -> String {
    log_store.to_uri(&object_store_path(decoded_path))
}

fn dv_to_kernel_descriptor(
    dv: &DeletionVectorDescriptor,
) -> delta_kernel::actions::deletion_vector::DeletionVectorDescriptor {
    let storage_type = match dv.storage_type {
        StorageType::UuidRelativePath => {
            delta_kernel::actions::deletion_vector::DeletionVectorStorageType::PersistedRelative
        }
        StorageType::Inline => {
            delta_kernel::actions::deletion_vector::DeletionVectorStorageType::Inline
        }
        StorageType::AbsolutePath => {
            delta_kernel::actions::deletion_vector::DeletionVectorStorageType::PersistedAbsolute
        }
    };
    delta_kernel::actions::deletion_vector::DeletionVectorDescriptor {
        storage_type,
        path_or_inline_dv: dv.path_or_inline_dv.clone(),
        offset: dv.offset,
        size_in_bytes: dv.size_in_bytes,
        cardinality: dv.cardinality,
    }
}

fn url_to_uri_string(url: Url) -> String {
    if url.scheme() == "file" {
        url.to_file_path()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| url.to_string())
    } else {
        url.to_string()
    }
}

fn dv_file_uri(
    dv: &delta_kernel::actions::deletion_vector::DeletionVectorDescriptor,
    root: &Url,
) -> Result<Option<String>, ArrowError> {
    dv.absolute_path(root)
        .map(|uri| uri.map(url_to_uri_string))
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))
}

pub(crate) fn deletion_vectors_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("file_uri", DataType::Utf8, false),
        Field::new("dv_storage_type", DataType::Utf8, true),
        Field::new("dv_path_or_inline_dv", DataType::Utf8, true),
        Field::new("dv_offset", DataType::Int32, true),
        Field::new("dv_size_in_bytes", DataType::Int32, true),
        Field::new("dv_cardinality", DataType::Int64, true),
        Field::new("dv_file_uri", DataType::Utf8, true),
        Field::new("dv_unique_id", DataType::Utf8, true),
    ]))
}

pub(crate) struct DeletionVectorsMetadataReader {
    schema: SchemaRef,
    log_store: LogStoreRef,
    table_root: Url,
    include_all_files: bool,
    batch_size: usize,
    files: Vec<LogicalFileView>,
    index: usize,
}

impl DeletionVectorsMetadataReader {
    pub(crate) fn new(
        log_store: LogStoreRef,
        state: deltalake::kernel::EagerSnapshot,
        include_all_files: bool,
        batch_size: usize,
    ) -> Self {
        let files = state.log_data().iter().collect::<Vec<_>>();
        Self {
            schema: deletion_vectors_schema(),
            table_root: table_root_url(&log_store),
            log_store,
            include_all_files,
            batch_size,
            files,
            index: 0,
        }
    }
}

impl Iterator for DeletionVectorsMetadataReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.files.len() {
            return None;
        }

        let mut path_builder = StringBuilder::with_capacity(self.batch_size, 1024);
        let mut file_uri_builder = StringBuilder::with_capacity(self.batch_size, 1024);

        let mut dv_storage_type_builder = StringBuilder::with_capacity(self.batch_size, 64);
        let mut dv_path_or_inline_dv_builder = StringBuilder::with_capacity(self.batch_size, 1024);
        let mut dv_offset_builder = Int32Builder::with_capacity(self.batch_size);
        let mut dv_size_in_bytes_builder = Int32Builder::with_capacity(self.batch_size);
        let mut dv_cardinality_builder = Int64Builder::with_capacity(self.batch_size);
        let mut dv_file_uri_builder = StringBuilder::with_capacity(self.batch_size, 1024);
        let mut dv_unique_id_builder = StringBuilder::with_capacity(self.batch_size, 256);

        let mut rows = 0usize;
        while rows < self.batch_size && self.index < self.files.len() {
            let file = self.files[self.index].clone();
            self.index += 1;

            let path = file.path().to_string();
            let uri = file_uri(&self.log_store, &path);

            let dv = file.deletion_vector_descriptor();
            if dv.is_none() && !self.include_all_files {
                continue;
            }

            path_builder.append_value(&path);
            file_uri_builder.append_value(&uri);

            match dv {
                Some(dv) => {
                    let kernel_dv = dv_to_kernel_descriptor(&dv);
                    let kernel_dv_file_uri = match dv_file_uri(&kernel_dv, &self.table_root) {
                        Ok(uri) => uri,
                        Err(e) => return Some(Err(e)),
                    };

                    dv_storage_type_builder.append_value(dv.storage_type.as_ref());
                    dv_path_or_inline_dv_builder.append_value(&dv.path_or_inline_dv);
                    dv_offset_builder.append_option(dv.offset);
                    dv_size_in_bytes_builder.append_value(dv.size_in_bytes);
                    dv_cardinality_builder.append_value(dv.cardinality);
                    match kernel_dv_file_uri {
                        Some(uri) => dv_file_uri_builder.append_value(&uri),
                        None => dv_file_uri_builder.append_null(),
                    }

                    let unique_id = kernel_dv.unique_id();
                    dv_unique_id_builder.append_value(&unique_id);
                }
                None => {
                    dv_storage_type_builder.append_null();
                    dv_path_or_inline_dv_builder.append_null();
                    dv_offset_builder.append_null();
                    dv_size_in_bytes_builder.append_null();
                    dv_cardinality_builder.append_null();
                    dv_file_uri_builder.append_null();
                    dv_unique_id_builder.append_null();
                }
            }

            rows += 1;
        }

        if rows == 0 {
            return None;
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(path_builder.finish()),
            Arc::new(file_uri_builder.finish()),
            Arc::new(dv_storage_type_builder.finish()),
            Arc::new(dv_path_or_inline_dv_builder.finish()),
            Arc::new(dv_offset_builder.finish()),
            Arc::new(dv_size_in_bytes_builder.finish()),
            Arc::new(dv_cardinality_builder.finish()),
            Arc::new(dv_file_uri_builder.finish()),
            Arc::new(dv_unique_id_builder.finish()),
        ];

        Some(RecordBatch::try_new(self.schema.clone(), columns))
    }
}

impl RecordBatchReader for DeletionVectorsMetadataReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
