#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_rows(
    fixtures_dir: Path, scale: str, limit: int = 5000
) -> list[dict[str, Any]]:
    path = fixtures_dir / scale / "narrow_sales" / "rows.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _approx_bytes(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return len(rows) * 48


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _expected_failure(rows: list[dict[str, Any]], message: str) -> dict[str, Any]:
    return {
        "rows_processed": len(rows),
        "bytes_processed": _approx_bytes(rows),
        "operations": 1,
        "table_version": None,
        "peak_rss_mb": None,
        "cpu_time_ms": None,
        "bytes_read": _approx_bytes(rows),
        "bytes_written": 0,
        "files_touched": None,
        "files_skipped": None,
        "spill_bytes": None,
        "result_hash": _hash_payload({"message": message, "rows": len(rows)}),
        "classification": "expected_failure",
    }


def _pandas_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        return _expected_failure(rows, f"missing dependency: {exc}")

    df = pd.DataFrame(rows)
    grouped = (
        df[df["flag"]]
        .groupby("region", as_index=False)["value_i64"]
        .sum()
        .sort_values("region")
    )
    payload = {
        "rows_processed": int(df.shape[0]),
        "bytes_processed": _approx_bytes(rows),
        "operations": 1,
        "table_version": None,
        "peak_rss_mb": None,
        "cpu_time_ms": None,
        "bytes_read": _approx_bytes(rows),
        "bytes_written": 0,
        "files_touched": None,
        "files_skipped": None,
        "spill_bytes": 0,
        "result_hash": _hash_payload(grouped.to_dict(orient="records")),
        "classification": "supported",
    }
    return payload


def _polars_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import polars as pl
    except ImportError as exc:
        return _expected_failure(rows, f"missing dependency: {exc}")

    frame = pl.DataFrame(rows)
    grouped = (
        frame.filter(pl.col("flag") == True)
        .group_by("region")
        .agg(pl.col("value_i64").sum().alias("value_i64_sum"))
        .sort("region")
    )
    return {
        "rows_processed": int(frame.height),
        "bytes_processed": _approx_bytes(rows),
        "operations": 1,
        "table_version": None,
        "peak_rss_mb": None,
        "cpu_time_ms": None,
        "bytes_read": _approx_bytes(rows),
        "bytes_written": 0,
        "files_touched": None,
        "files_skipped": None,
        "spill_bytes": 0,
        "result_hash": _hash_payload(grouped.to_dicts()),
        "classification": "supported",
    }


def _pyarrow_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as exc:
        return _expected_failure(rows, f"missing dependency: {exc}")

    table = pa.table(
        {
            "id": [row["id"] for row in rows],
            "flag": [row["flag"] for row in rows],
            "value_i64": [row["value_i64"] for row in rows],
        }
    )
    mask = pc.and_(
        pc.equal(table["flag"], pa.scalar(True)),
        pc.greater(table["value_i64"], pa.scalar(0)),
    )
    filtered = table.filter(mask)
    result_value = int(pc.sum(filtered["value_i64"]).as_py() or 0)

    return {
        "rows_processed": int(table.num_rows),
        "bytes_processed": _approx_bytes(rows),
        "operations": 1,
        "table_version": None,
        "peak_rss_mb": None,
        "cpu_time_ms": None,
        "bytes_read": _approx_bytes(rows),
        "bytes_written": 0,
        "files_touched": None,
        "files_skipped": None,
        "spill_bytes": 0,
        "result_hash": _hash_payload({"sum": result_value, "rows": filtered.num_rows}),
        "classification": "supported",
    }


def _run_case(case: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if case == "pandas_roundtrip_smoke":
        return _pandas_case(rows)
    if case == "polars_roundtrip_smoke":
        return _polars_case(rows)
    if case == "pyarrow_dataset_scan_perf":
        return _pyarrow_case(rows)
    raise ValueError(f"unknown case: {case}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run delta-bench python interop case")
    parser.add_argument("--case", required=True)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args()

    rows = _load_rows(Path(args.fixtures_dir), args.scale)
    if not rows:
        raise SystemExit("no rows loaded from fixture set")

    result = _run_case(args.case, rows)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
