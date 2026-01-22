import os

import pytest
from arro3.core import Array, DataType, Field, Table

from deltalake import DeltaTable, write_deltalake


@pytest.mark.datafusion
def test_datafusion_table_provider(tmp_path):
    if os.environ.get("DELTALAKE_RUN_DATAFUSION_TESTS") != "1":
        pytest.skip(
            "DataFusion Python integration tests are disabled by default; set DELTALAKE_RUN_DATAFUSION_TESTS=1"
        )

    try:
        from datafusion import SessionContext
    except ImportError:
        pytest.skip("DataFusion Python is not installed")
    nrows = 5
    table = Table(
        {
            "id": Array(
                ["1", "2", "3", "4", "5"],
                Field("id", type=DataType.string_view(), nullable=True),
            ),
            "price": Array(
                list(range(nrows)), Field("price", type=DataType.int64(), nullable=True)
            ),
            "sold": Array(
                list(range(nrows)), Field("sold", type=DataType.int32(), nullable=True)
            ),
            "deleted": Array(
                [False] * nrows, Field("deleted", type=DataType.bool(), nullable=True)
            ),
        },
    )

    write_deltalake(tmp_path, table)

    dt = DeltaTable(tmp_path)

    session = SessionContext()
    try:
        session.register_table("tbl", dt)
    except ImportError as err:
        msg = str(err)
        if "Incompatible libraries" in msg and "table providers" in msg:
            pytest.skip(msg)
        raise

    data = session.sql("SELECT * FROM tbl")

    assert Table.from_arrow(data) == table
