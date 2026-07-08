import pathlib
from typing import Any
from urllib.parse import unquote, urlparse

from deltalake import DeltaTable


def active_parquet_path(dt: DeltaTable) -> pathlib.Path:
    file_uris = dt.file_uris()
    assert len(file_uris) == 1
    uri = file_uris[0]
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return pathlib.Path(unquote(parsed.path))
    assert parsed.scheme == ""
    return pathlib.Path(uri)


def assert_delta_parquet_contract(
    parquet_file: Any,
    *,
    string_field: str | None = None,
    compression: str | None = None,
) -> None:
    metadata_keys = set((parquet_file.metadata.metadata or {}).keys())
    assert b"ARROW:schema" not in metadata_keys

    if string_field is not None:
        import pyarrow as pa

        assert parquet_file.schema_arrow.field(string_field).type == pa.string()

    if compression is not None:
        compressions = {
            parquet_file.metadata.row_group(row_group).column(column).compression
            for row_group in range(parquet_file.metadata.num_row_groups)
            for column in range(parquet_file.metadata.row_group(row_group).num_columns)
        }
        assert compressions == {compression}
