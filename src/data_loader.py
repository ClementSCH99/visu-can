from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import pandas as pd

from .summary import DatasetSummary, compute_dataset_summary


@dataclass(frozen=True)
class SignalSummary:
    name: str
    samples: int


@dataclass(frozen=True)
class PlotTrace:
    name: str
    x: list[float]
    y: list[float]


@dataclass(frozen=True)
class SignalIndex:
    processed_rows: int
    signal_counts: dict[str, int]
    row_ids_by_signal: dict[str, list[int]]
    timestamps_by_signal: dict[str, list[float]]
    values_by_signal: dict[str, list[float]]
    first_timestamp: float | None
    last_timestamp: float | None


@dataclass(frozen=True)
class PlotDataset:
    traces: list[PlotTrace]
    processed_rows: int
    matched_rows: int
    first_timestamp: float | None
    last_timestamp: float | None


_CSV_METADATA_COLUMNS = frozenset({"can_id", "dlc", "data_hex"})
# Columns tried in order; first match becomes the time axis
_TIMESTAMP_CANDIDATES = ("timestamp", "timestamp_s", "nhr_monotonic_s", "nhr_timestamp_utc")
_SIGNAL_SUFFIX_EXCLUDE = ("_age_s", "_status")


def _build_signal_index_from_csv(file_path: str | Path, row_limit: int | None = None) -> "SignalIndex":
    df = pd.read_csv(file_path, nrows=row_limit)
    total_rows = len(df)

    empty = SignalIndex(
        processed_rows=total_rows,
        signal_counts={},
        row_ids_by_signal={},
        timestamps_by_signal={},
        values_by_signal={},
        first_timestamp=None,
        last_timestamp=None,
    )
    if total_rows == 0:
        return empty

    ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        return empty

    if ts_col == "nhr_timestamp_utc":
        parsed_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce", format="ISO8601")
        ts_values = (parsed_ts.dt.as_unit("ns").astype("int64") / 1e9).where(parsed_ts.notna())
    else:
        ts_values = pd.to_numeric(df[ts_col], errors="coerce")

    # Keep timestamps separate: adding a column to a wide CSV frame fragments it.
    timestamps = ts_values.to_numpy(dtype=float)
    valid_rows = np.flatnonzero(np.isfinite(timestamps))
    sorted_rows = valid_rows[np.argsort(timestamps[valid_rows], kind="stable")]
    df = df.iloc[sorted_rows].reset_index(drop=True)
    if df.empty:
        return empty

    ts_array = timestamps[sorted_rows]
    ts_array = ts_array - ts_array[0]
    row_ids_base = np.arange(1, len(df) + 1, dtype=np.int64)
    meta_cols = _CSV_METADATA_COLUMNS | {"_ts", ts_col}
    signal_cols = [
        col for col in df.columns
        if col not in meta_cols
        and not any(col.endswith(s) for s in _SIGNAL_SUFFIX_EXCLUDE)
    ]

    signal_counts: dict[str, int] = {}
    row_ids_by_signal: dict[str, list[int]] = {}
    timestamps_by_signal: dict[str, list[float]] = {}
    values_by_signal: dict[str, list[float]] = {}

    for col in signal_cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        indices = np.flatnonzero(numeric_col.notna().to_numpy())
        if len(indices) == 0:
            continue
        signal_counts[col] = len(indices)
        row_ids_by_signal[col] = row_ids_base[indices].tolist()
        timestamps_by_signal[col] = ts_array[indices].tolist()
        values_by_signal[col] = numeric_col.to_numpy(dtype=float, na_value=np.nan)[indices].tolist()

    return SignalIndex(
        processed_rows=total_rows,
        signal_counts=signal_counts,
        row_ids_by_signal=row_ids_by_signal,
        timestamps_by_signal=timestamps_by_signal,
        values_by_signal=values_by_signal,
        first_timestamp=float(ts_array[0]),
        last_timestamp=float(ts_array[-1]),
    )


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _build_signal_index_from_ndjson(file_path: str | Path, row_limit: int | None = None) -> SignalIndex:
    timestamps: list[float] = []
    parsed_rows: list[dict] = []
    processed_rows = 0

    with Path(file_path).open("rb") as handle:
        for index, line in enumerate(handle):
            if row_limit is not None and index >= row_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = orjson.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            processed_rows += 1
            ts = row.get("timestamp")
            parsed = row.get("parsed")
            if not isinstance(parsed, dict) or not _is_numeric(ts) or not math.isfinite(float(ts)):
                continue
            timestamps.append(float(cast(int | float, ts)))
            parsed_rows.append(parsed)

    if not timestamps:
        return SignalIndex(
            processed_rows=processed_rows,
            signal_counts={},
            row_ids_by_signal={},
            timestamps_by_signal={},
            values_by_signal={},
            first_timestamp=None,
            last_timestamp=None,
        )

    ts_array = np.asarray(timestamps, dtype=float)
    order = np.argsort(ts_array, kind="stable")
    ts_array = ts_array[order]
    ts_array = ts_array - ts_array[0]
    row_ids_base = np.arange(1, len(timestamps) + 1, dtype=np.int64)
    parsed_df = pd.DataFrame(parsed_rows).iloc[order].reset_index(drop=True)

    signal_counts: dict[str, int] = {}
    row_ids_by_signal: dict[str, list[int]] = {}
    timestamps_by_signal: dict[str, list[float]] = {}
    values_by_signal: dict[str, list[float]] = {}

    for col in parsed_df.columns:
        numeric_col = pd.to_numeric(parsed_df[col], errors="coerce")
        indices = np.flatnonzero(numeric_col.notna().to_numpy())
        if len(indices) == 0:
            continue
        signal_counts[col] = len(indices)
        row_ids_by_signal[col] = row_ids_base[indices].tolist()
        timestamps_by_signal[col] = ts_array[indices].tolist()
        values_by_signal[col] = numeric_col.to_numpy(dtype=float, na_value=np.nan)[indices].tolist()

    return SignalIndex(
        processed_rows=processed_rows,
        signal_counts=signal_counts,
        row_ids_by_signal=row_ids_by_signal,
        timestamps_by_signal=timestamps_by_signal,
        values_by_signal=values_by_signal,
        first_timestamp=float(ts_array[0]),
        last_timestamp=float(ts_array[-1]),
    )


def build_signal_index(file_path: str | Path, row_limit: int | None = None) -> SignalIndex:
    if Path(file_path).suffix.lower() == ".csv":
        return _build_signal_index_from_csv(file_path, row_limit=row_limit)
    return _build_signal_index_from_ndjson(file_path, row_limit=row_limit)


def discover_numeric_signals(file_path: str | Path, scan_limit: int | None = 50000) -> list[SignalSummary]:
    signal_index = build_signal_index(file_path, row_limit=scan_limit)
    return [
        SignalSummary(name=name, samples=signal_index.signal_counts[name])
        for name in sorted(
            signal_index.signal_counts,
            key=lambda item: (-signal_index.signal_counts[item], item.lower()),
        )
    ]


def load_selected_signals(
    file_path: str | Path,
    selected_signals: list[str],
    row_limit: int | None = None,
    max_points_per_signal: int = 5000,
    start_time: float | None = None,
    end_time: float | None = None,
    sample_every: int = 1,
    signal_index: SignalIndex | None = None,
) -> PlotDataset:
    active_index = signal_index or build_signal_index(file_path, row_limit=row_limit)
    traces: list[PlotTrace] = []
    matched_row_ids: set[int] = set()
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    for signal_name in selected_signals:
        timestamps = active_index.timestamps_by_signal.get(signal_name)
        values = active_index.values_by_signal.get(signal_name)
        row_ids = active_index.row_ids_by_signal.get(signal_name)
        if not timestamps or not values or not row_ids:
            continue

        start_index = bisect_left(timestamps, start_time) if start_time is not None else 0
        end_index = bisect_right(timestamps, end_time) if end_time is not None else len(timestamps)
        if start_index >= end_index:
            continue

        windowed_timestamps = timestamps[start_index:end_index]
        windowed_values = values[start_index:end_index]
        matched_row_ids.update(row_ids[start_index:end_index])

        if first_timestamp is None or windowed_timestamps[0] < first_timestamp:
            first_timestamp = windowed_timestamps[0]
        if last_timestamp is None or windowed_timestamps[-1] > last_timestamp:
            last_timestamp = windowed_timestamps[-1]

        sampled_timestamps, sampled_values = sample_trace(
            windowed_timestamps,
            windowed_values,
            sample_every,
            max_points_per_signal,
        )

        traces.append(
            PlotTrace(
                name=signal_name,
                x=sampled_timestamps,
                y=sampled_values,
            )
        )

    return PlotDataset(
        traces=traces,
        processed_rows=active_index.processed_rows,
        matched_rows=len(matched_row_ids),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
    )


def sample_by_stride(values: list[float], sample_every: int) -> list[float]:
    if sample_every <= 1:
        return values
    sampled = values[::sample_every]
    if sampled and sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def sample_trace(
    timestamps: list[float],
    values: list[float],
    sample_every: int,
    max_points: int,
) -> tuple[list[float], list[float]]:
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have the same length")

    sampled_indices = build_sample_indices(len(timestamps), sample_every, max_points)
    if not sampled_indices:
        return [], []
    idx = np.asarray(sampled_indices)
    return np.asarray(timestamps)[idx].tolist(), np.asarray(values)[idx].tolist()


def build_sample_indices(length: int, sample_every: int, max_points: int) -> list[int]:
    if length <= 0:
        return []

    stride_indices = list(range(0, length, max(1, sample_every)))
    if stride_indices[-1] != length - 1:
        stride_indices.append(length - 1)

    if max_points <= 0 or len(stride_indices) <= max_points:
        return stride_indices

    if max_points == 1:
        return [stride_indices[0]]
    last_position = len(stride_indices) - 1
    return [stride_indices[position * last_position // (max_points - 1)] for position in range(max_points)]


def downsample_series(values: list[float], max_points: int) -> list[float]:
    if max_points <= 0 or len(values) <= max_points:
        return values
    stride = max(1, math.ceil(len(values) / max_points))
    sampled = values[::stride]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled
