use std::sync::Arc;

use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use deltalake::arrow::array::{
    ArrayRef, AsArray, Int32Builder, Int64Builder, LargeBinaryBuilder, RecordBatchReader,
    StringBuilder,
};
use deltalake::arrow::datatypes::{Int32Type, Int64Type};
use deltalake::arrow::record_batch::RecordBatch;
use deltalake::kernel::{DeletionVectorDescriptor, LogicalFileView, StorageType};
use deltalake::logstore::LogStoreRef;
use futures::{StreamExt as _, stream};
use object_store::path::Path as ObjectStorePath;
use url::Url;

const DV_ROARING_MAGIC: u32 = 1681511377;

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

pub(crate) fn deletion_vector_roaring_bytes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("file_uri", DataType::Utf8, false),
        Field::new("dv_unique_id", DataType::Utf8, true),
        Field::new("dv_size_in_bytes", DataType::Int32, true),
        Field::new("dv_roaring_bytes", DataType::LargeBinary, true),
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

#[derive(Debug)]
struct DvPayloadError {
    dv_unique_id: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for DvPayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed to read deletion vector {}: {}",
            self.dv_unique_id, self.source
        )
    }
}

impl std::error::Error for DvPayloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

#[derive(Clone, Debug)]
struct DvJob {
    row: usize,
    dv_unique_id: String,
    expected_size_in_bytes: i32,
    dv: delta_kernel::actions::deletion_vector::DeletionVectorDescriptor,
}

struct ChunkedRecordBatchReader {
    reader: Box<dyn RecordBatchReader + Send>,
    batch_size: usize,
    pending: Option<RecordBatch>,
    offset: usize,
}

impl ChunkedRecordBatchReader {
    fn new(reader: Box<dyn RecordBatchReader + Send>, batch_size: usize) -> Self {
        Self {
            reader,
            batch_size,
            pending: None,
            offset: 0,
        }
    }
}

impl Iterator for ChunkedRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.pending.take() {
                Some(batch) => batch,
                None => match self.reader.next()? {
                    Ok(batch) => batch,
                    Err(e) => return Some(Err(e)),
                },
            };

            if self.offset >= batch.num_rows() {
                self.offset = 0;
                continue;
            }

            let take_len = (batch.num_rows() - self.offset).min(self.batch_size);
            let out = batch.slice(self.offset, take_len);
            self.offset += take_len;

            if self.offset < batch.num_rows() {
                self.pending = Some(batch);
            } else {
                self.offset = 0;
            }

            return Some(Ok(out));
        }
    }
}

impl RecordBatchReader for ChunkedRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.reader.schema()
    }
}

pub(crate) struct DeletionVectorRoaringBytesReader {
    schema: SchemaRef,
    input: ChunkedRecordBatchReader,
    storage: Arc<dyn delta_kernel::StorageHandler>,
    table_root: Url,
    max_concurrent: usize,
    path_idx: usize,
    file_uri_idx: usize,
    dv_storage_type_idx: usize,
    dv_path_or_inline_dv_idx: usize,
    dv_offset_idx: usize,
    dv_size_in_bytes_idx: usize,
    dv_cardinality_idx: usize,
    dv_unique_id_idx: Option<usize>,
}

fn is_string_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn schema_error(
    schema: &Schema,
    missing: Vec<&'static str>,
    wrong_types: Vec<(&'static str, DataType)>,
) -> ArrowError {
    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!("missing columns: {}", missing.join(", ")));
    }
    if !wrong_types.is_empty() {
        let items = wrong_types
            .into_iter()
            .map(|(name, dt)| format!("{name}: {dt:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("wrong types: {items}"));
    }

    let expected = [
        "path: utf8",
        "file_uri: utf8",
        "dv_storage_type: utf8?",
        "dv_path_or_inline_dv: utf8?",
        "dv_offset: int32?",
        "dv_size_in_bytes: int32?",
        "dv_cardinality: int64?",
        "dv_unique_id: utf8? (optional)",
    ]
    .join(", ");

    ArrowError::SchemaError(format!(
        "Invalid deletion vector descriptor batch schema (expected {expected}). {}. Input schema: {:?}",
        parts.join("; "),
        schema
    ))
}

impl DeletionVectorRoaringBytesReader {
    pub(crate) fn try_new(
        input: Box<dyn RecordBatchReader + Send>,
        storage: Arc<dyn delta_kernel::StorageHandler>,
        table_root: Url,
        batch_size: usize,
        max_concurrent: usize,
    ) -> Result<Self, ArrowError> {
        let schema = input.schema();

        let mut missing = Vec::new();
        let mut wrong_types = Vec::new();

        let mut idx = |name: &'static str| match schema.index_of(name) {
            Ok(i) => Some(i),
            Err(_) => {
                missing.push(name);
                None
            }
        };

        let path_idx = idx("path");
        let file_uri_idx = idx("file_uri");
        let dv_storage_type_idx = idx("dv_storage_type");
        let dv_path_or_inline_dv_idx = idx("dv_path_or_inline_dv");
        let dv_offset_idx = idx("dv_offset");
        let dv_size_in_bytes_idx = idx("dv_size_in_bytes");
        let dv_cardinality_idx = idx("dv_cardinality");
        let dv_unique_id_idx = schema.index_of("dv_unique_id").ok();

        if let Some(i) = path_idx {
            if !is_string_type(schema.field(i).data_type()) {
                wrong_types.push(("path", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = file_uri_idx {
            if !is_string_type(schema.field(i).data_type()) {
                wrong_types.push(("file_uri", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_storage_type_idx {
            if !is_string_type(schema.field(i).data_type()) {
                wrong_types.push(("dv_storage_type", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_path_or_inline_dv_idx {
            if !is_string_type(schema.field(i).data_type()) {
                wrong_types.push(("dv_path_or_inline_dv", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_offset_idx {
            if schema.field(i).data_type() != &DataType::Int32 {
                wrong_types.push(("dv_offset", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_size_in_bytes_idx {
            if schema.field(i).data_type() != &DataType::Int32 {
                wrong_types.push(("dv_size_in_bytes", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_cardinality_idx {
            if schema.field(i).data_type() != &DataType::Int64 {
                wrong_types.push(("dv_cardinality", schema.field(i).data_type().clone()));
            }
        }
        if let Some(i) = dv_unique_id_idx {
            if !is_string_type(schema.field(i).data_type()) {
                wrong_types.push(("dv_unique_id", schema.field(i).data_type().clone()));
            }
        }

        if !missing.is_empty() || !wrong_types.is_empty() {
            return Err(schema_error(schema.as_ref(), missing, wrong_types));
        }

        Ok(Self {
            schema: deletion_vector_roaring_bytes_schema(),
            input: ChunkedRecordBatchReader::new(input, batch_size),
            storage,
            table_root,
            max_concurrent,
            path_idx: path_idx.unwrap(),
            file_uri_idx: file_uri_idx.unwrap(),
            dv_storage_type_idx: dv_storage_type_idx.unwrap(),
            dv_path_or_inline_dv_idx: dv_path_or_inline_dv_idx.unwrap(),
            dv_offset_idx: dv_offset_idx.unwrap(),
            dv_size_in_bytes_idx: dv_size_in_bytes_idx.unwrap(),
            dv_cardinality_idx: dv_cardinality_idx.unwrap(),
            dv_unique_id_idx,
        })
    }
}

impl Iterator for DeletionVectorRoaringBytesReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.input.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e)),
        };

        let schema = batch.schema();
        let row_count = batch.num_rows();

        let path_col = batch.column(self.path_idx).as_ref();
        let file_uri_col = batch.column(self.file_uri_idx).as_ref();
        let storage_type_col = batch.column(self.dv_storage_type_idx).as_ref();
        let path_or_inline_col = batch.column(self.dv_path_or_inline_dv_idx).as_ref();
        let offset_col = batch.column(self.dv_offset_idx).as_ref();
        let size_col = batch.column(self.dv_size_in_bytes_idx).as_ref();
        let cardinality_col = batch.column(self.dv_cardinality_idx).as_ref();

        let unique_id_col = self
            .dv_unique_id_idx
            .and_then(|idx| batch.schema().index_of("dv_unique_id").ok().map(|_| idx))
            .map(|idx| batch.column(idx).as_ref());

        let mut unique_id_by_row: Vec<Option<String>> = vec![None; row_count];
        let mut size_by_row: Vec<Option<i32>> = vec![None; row_count];
        let mut jobs: Vec<DvJob> = Vec::new();

        for row in 0..row_count {
            if !storage_type_col.is_valid(row) {
                continue;
            }

            if !path_or_inline_col.is_valid(row) {
                return Some(Err(ArrowError::SchemaError(
                    "dv_path_or_inline_dv must be non-null when dv_storage_type is set".into(),
                )));
            }

            let storage_type = match storage_type_col.data_type() {
                DataType::Utf8 => storage_type_col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => storage_type_col.as_string::<i64>().value(row),
                DataType::Utf8View => storage_type_col.as_string_view().value(row),
                other => {
                    return Some(Err(ArrowError::SchemaError(format!(
                        "Invalid dv_storage_type type {other:?} in batch schema {schema:?}"
                    ))));
                }
            };

            let path_or_inline = match path_or_inline_col.data_type() {
                DataType::Utf8 => path_or_inline_col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => path_or_inline_col.as_string::<i64>().value(row),
                DataType::Utf8View => path_or_inline_col.as_string_view().value(row),
                other => {
                    return Some(Err(ArrowError::SchemaError(format!(
                        "Invalid dv_path_or_inline_dv type {other:?} in batch schema {schema:?}"
                    ))));
                }
            };

            if !size_col.is_valid(row) || !cardinality_col.is_valid(row) {
                return Some(Err(ArrowError::SchemaError(
                    "DV descriptor columns must be non-null when dv_storage_type is set".into(),
                )));
            }

            let offset = if offset_col.is_valid(row) {
                Some(offset_col.as_primitive::<Int32Type>().value(row))
            } else {
                None
            };
            let size_in_bytes = size_col.as_primitive::<Int32Type>().value(row);
            let cardinality = cardinality_col.as_primitive::<Int64Type>().value(row);

            let parsed_storage_type =
                match storage_type
                    .parse::<delta_kernel::actions::deletion_vector::DeletionVectorStorageType>()
                {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(ArrowError::ExternalError(Box::new(e))));
                    }
                };

            let dv = delta_kernel::actions::deletion_vector::DeletionVectorDescriptor {
                storage_type: parsed_storage_type,
                path_or_inline_dv: path_or_inline.to_string(),
                offset,
                size_in_bytes,
                cardinality,
            };

            let unique_id = match unique_id_col {
                Some(col) if col.is_valid(row) => match col.data_type() {
                    DataType::Utf8 => col.as_string::<i32>().value(row).to_string(),
                    DataType::LargeUtf8 => col.as_string::<i64>().value(row).to_string(),
                    DataType::Utf8View => col.as_string_view().value(row).to_string(),
                    other => {
                        return Some(Err(ArrowError::SchemaError(format!(
                            "Invalid dv_unique_id type {other:?} in batch schema {schema:?}"
                        ))));
                    }
                },
                _ => dv.unique_id(),
            };

            unique_id_by_row[row] = Some(unique_id.clone());
            size_by_row[row] = Some(size_in_bytes);
            jobs.push(DvJob {
                row,
                dv_unique_id: unique_id,
                expected_size_in_bytes: size_in_bytes,
                dv,
            });
        }

        let storage = self.storage.clone();
        let root = self.table_root.clone();
        let max_concurrent = self.max_concurrent;

        let mut roaring_bytes_by_row: Vec<Option<Vec<u8>>> = vec![None; row_count];
        let results = crate::utils::rt().block_on(async {
            stream::iter(jobs)
                .map(|job| {
                    let storage = storage.clone();
                    let root = root.clone();
                    async move {
                        tokio::task::spawn_blocking(move || {
                            let treemap = job.dv.read(storage, &root).map_err(|e| {
                                ArrowError::ExternalError(Box::new(DvPayloadError {
                                    dv_unique_id: job.dv_unique_id.clone(),
                                    source: Box::new(e),
                                }))
                            })?;

                            let mut bytes = Vec::with_capacity(4 + treemap.serialized_size());
                            bytes.extend_from_slice(&DV_ROARING_MAGIC.to_le_bytes());
                            treemap.serialize_into(&mut bytes).map_err(|e| {
                                ArrowError::ExternalError(Box::new(DvPayloadError {
                                    dv_unique_id: job.dv_unique_id.clone(),
                                    source: Box::new(e),
                                }))
                            })?;

                            if bytes.len() != job.expected_size_in_bytes as usize {
                                return Err(ArrowError::ExternalError(Box::new(DvPayloadError {
                                    dv_unique_id: job.dv_unique_id.clone(),
                                    source: Box::new(std::io::Error::new(
                                        std::io::ErrorKind::InvalidData,
                                        format!(
                                            "Size mismatch (expected {}, got {})",
                                            job.expected_size_in_bytes,
                                            bytes.len()
                                        ),
                                    )),
                                })));
                            }

                            Ok::<_, ArrowError>((job.row, bytes))
                        })
                        .await
                        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?
                    }
                })
                .buffer_unordered(max_concurrent)
                .collect::<Vec<Result<(usize, Vec<u8>), ArrowError>>>()
                .await
        });

        for res in results {
            match res {
                Ok((row, bytes)) => roaring_bytes_by_row[row] = Some(bytes),
                Err(e) => return Some(Err(e)),
            }
        }

        let mut out_path = StringBuilder::with_capacity(row_count, 1024);
        let mut out_file_uri = StringBuilder::with_capacity(row_count, 1024);
        let mut out_unique_id = StringBuilder::with_capacity(row_count, 256);
        let mut out_size_in_bytes = Int32Builder::with_capacity(row_count);
        let mut out_bytes = LargeBinaryBuilder::with_capacity(row_count, 1024);

        for row in 0..row_count {
            if !path_col.is_valid(row) || !file_uri_col.is_valid(row) {
                return Some(Err(ArrowError::SchemaError(
                    "path and file_uri must be non-null in deletion vector descriptor batches"
                        .into(),
                )));
            }

            let path = match path_col.data_type() {
                DataType::Utf8 => path_col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => path_col.as_string::<i64>().value(row),
                DataType::Utf8View => path_col.as_string_view().value(row),
                other => {
                    return Some(Err(ArrowError::SchemaError(format!(
                        "Invalid path type {other:?} in batch schema {schema:?}"
                    ))));
                }
            };
            let file_uri = match file_uri_col.data_type() {
                DataType::Utf8 => file_uri_col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => file_uri_col.as_string::<i64>().value(row),
                DataType::Utf8View => file_uri_col.as_string_view().value(row),
                other => {
                    return Some(Err(ArrowError::SchemaError(format!(
                        "Invalid file_uri type {other:?} in batch schema {schema:?}"
                    ))));
                }
            };

            out_path.append_value(path);
            out_file_uri.append_value(file_uri);

            match unique_id_by_row[row].as_deref() {
                Some(v) => out_unique_id.append_value(v),
                None => out_unique_id.append_null(),
            }
            out_size_in_bytes.append_option(size_by_row[row]);

            match roaring_bytes_by_row[row].as_deref() {
                Some(b) => out_bytes.append_value(b),
                None => out_bytes.append_null(),
            }
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(out_path.finish()),
            Arc::new(out_file_uri.finish()),
            Arc::new(out_unique_id.finish()),
            Arc::new(out_size_in_bytes.finish()),
            Arc::new(out_bytes.finish()),
        ];

        Some(RecordBatch::try_new(self.schema.clone(), columns))
    }
}

impl RecordBatchReader for DeletionVectorRoaringBytesReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use bytes::Bytes;
    use delta_kernel::actions::deletion_vector::{
        DeletionVectorDescriptor, DeletionVectorStorageType,
    };
    use delta_kernel::{DeltaResult, Error, FileMeta, FileSlice, StorageHandler};
    use roaring::RoaringTreemap;
    use url::Url;

    #[derive(Debug)]
    struct DummyStorage;

    impl StorageHandler for DummyStorage {
        fn list_from(
            &self,
            _path: &Url,
        ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<FileMeta>>>> {
            Err(Error::internal_error("DummyStorage should not be used"))
        }

        fn read_files(
            &self,
            _files: Vec<FileSlice>,
        ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<Bytes>>>> {
            Err(Error::internal_error("DummyStorage should not be used"))
        }

        fn copy_atomic(&self, _src: &Url, _dest: &Url) -> DeltaResult<()> {
            Err(Error::internal_error("DummyStorage should not be used"))
        }

        fn head(&self, _path: &Url) -> DeltaResult<FileMeta> {
            Err(Error::internal_error("DummyStorage should not be used"))
        }
    }

    #[test]
    fn test_inline_dv_portable_roaring_bytes_roundtrip() {
        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::Inline,
            // Copied from delta-kernel tests; represents deletions for [3, 4, 7, 11, 18, 29]
            path_or_inline_dv: "^Bg9^0rr910000000000iXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L"
                .to_string(),
            offset: None,
            size_in_bytes: 44,
            cardinality: 6,
        };

        let storage: Arc<dyn StorageHandler> = Arc::new(DummyStorage);
        let parent = Url::parse("http://not.used").unwrap();
        let treemap = dv.read(storage, &parent).unwrap();

        let mut bytes = Vec::with_capacity(4 + treemap.serialized_size());
        bytes.extend_from_slice(&super::DV_ROARING_MAGIC.to_le_bytes());
        treemap.serialize_into(&mut bytes).unwrap();

        assert_eq!(bytes.len(), dv.size_in_bytes as usize);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            super::DV_ROARING_MAGIC
        );

        let roundtrip = RoaringTreemap::deserialize_from(Cursor::new(&bytes[4..])).unwrap();
        assert_eq!(treemap, roundtrip);
    }
}
