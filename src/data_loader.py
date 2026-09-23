from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import orjson
import pandas as pd


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


@dataclass(frozen=True)
class DatasetSummary:
    charged_capacity_ah: float | None
    discharged_capacity_ah: float | None
    charged_energy_kwh: float | None
    discharged_energy_kwh: float | None
    avg_temp_start_c: float | None
    avg_max_temp_c: float | None
    min_cell_voltage_v: float | None
    min_cell_voltage_id: str | None
    max_cell_voltage_v: float | None
    max_cell_voltage_id: str | None
    initial_cell_imbalance_mv: float | None


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

    df = df.assign(_ts=ts_values)
    df = df[np.isfinite(df["_ts"])].sort_values("_ts", kind="stable").reset_index(drop=True)
    if df.empty:
        return empty

    ts_array = df["_ts"].to_numpy(dtype=float)
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


_CELL_TEMP_PREFIX = "can_CELL_T_"
_CELL_VOLT_PREFIX = "can_CELL_V_"
_START_WINDOW_S = 60.0


def compute_dataset_summary(file_path: str | Path) -> "DatasetSummary | None":
    path = Path(file_path)
    if path.suffix.lower() != ".csv":
        return None

    df = pd.read_csv(file_path)
    if df.empty:
        return None

    ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        return None

    if ts_col == "nhr_timestamp_utc":
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").astype("int64") / 1e9
    else:
        ts = pd.to_numeric(df[ts_col], errors="coerce")

    first_valid_ts = ts.dropna().iloc[0] if not ts.dropna().empty else 0.0
    ts = ts - first_valid_ts

    # --- Capacity ---
    charged_capacity_ah: float | None = None
    discharged_capacity_ah: float | None = None

    if "nhr_capacity_charge_ah" in df.columns:
        col = pd.to_numeric(df["nhr_capacity_charge_ah"], errors="coerce").dropna()
        if not col.empty:
            charged_capacity_ah = float(col.iloc[-1] - col.iloc[0])

    if "nhr_capacity_discharge_ah" in df.columns:
        col = pd.to_numeric(df["nhr_capacity_discharge_ah"], errors="coerce").dropna()
        if not col.empty:
            discharged_capacity_ah = float(col.iloc[-1] - col.iloc[0])

    if (charged_capacity_ah is None or discharged_capacity_ah is None) and "nhr_current_a" in df.columns:
        current = pd.to_numeric(df["nhr_current_a"], errors="coerce")
        valid_mask = current.notna() & ts.notna()
        if valid_mask.sum() > 1:
            t = ts[valid_mask].to_numpy(dtype=float)
            i = current[valid_mask].to_numpy(dtype=float)
            if charged_capacity_ah is None:
                charged_capacity_ah = float(np.trapezoid(np.where(i > 0, i, 0.0), t) / 3600)
            if discharged_capacity_ah is None:
                discharged_capacity_ah = float(np.trapezoid(np.where(i < 0, -i, 0.0), t) / 3600)

    # --- Energy ---
    charged_energy_kwh: float | None = None
    discharged_energy_kwh: float | None = None

    if "nhr_energy_charge_kwh" in df.columns:
        col = pd.to_numeric(df["nhr_energy_charge_kwh"], errors="coerce").dropna()
        if not col.empty:
            charged_energy_kwh = float(col.iloc[-1] - col.iloc[0])

    if "nhr_energy_discharge_kwh" in df.columns:
        col = pd.to_numeric(df["nhr_energy_discharge_kwh"], errors="coerce").dropna()
        if not col.empty:
            discharged_energy_kwh = float(col.iloc[-1] - col.iloc[0])

    if (charged_energy_kwh is None or discharged_energy_kwh is None) and "nhr_power_w" in df.columns:
        power = pd.to_numeric(df["nhr_power_w"], errors="coerce")
        valid_mask = power.notna() & ts.notna()
        if valid_mask.sum() > 1:
            t = ts[valid_mask].to_numpy(dtype=float)
            p = power[valid_mask].to_numpy(dtype=float)
            if charged_energy_kwh is None:
                charged_energy_kwh = float(np.trapezoid(np.where(p > 0, p, 0.0), t) / 3_600_000)
            if discharged_energy_kwh is None:
                discharged_energy_kwh = float(np.trapezoid(np.where(p < 0, -p, 0.0), t) / 3_600_000)

    # --- Temperature (CAN cell sensors; fall back to NHR instrument if absent) ---
    temp_cols = [
        c for c in df.columns
        if c.startswith(_CELL_TEMP_PREFIX)
        and not any(c.endswith(s) for s in _SIGNAL_SUFFIX_EXCLUDE)
    ]
    if not temp_cols and "nhr_temperature_c" in df.columns:
        temp_cols = ["nhr_temperature_c"]

    avg_temp_start_c: float | None = None
    avg_max_temp_c: float | None = None

    if temp_cols:
        temp_df = df[temp_cols].apply(pd.to_numeric, errors="coerce")
        start_mask = ts <= _START_WINDOW_S

        start_vals = temp_df[start_mask].to_numpy().flatten()
        start_vals = start_vals[~np.isnan(start_vals)]
        if len(start_vals) > 0:
            avg_temp_start_c = float(np.mean(start_vals))

        col_maxes = temp_df.max(skipna=True).dropna()
        if not col_maxes.empty:
            avg_max_temp_c = float(col_maxes.mean())

    # --- Cell voltages ---
    cell_v_cols = [
        c for c in df.columns
        if c.startswith(_CELL_VOLT_PREFIX)
        and not any(c.endswith(s) for s in _SIGNAL_SUFFIX_EXCLUDE)
    ]

    min_cell_voltage_v: float | None = None
    min_cell_voltage_id: str | None = None
    max_cell_voltage_v: float | None = None
    max_cell_voltage_id: str | None = None
    initial_cell_imbalance_mv: float | None = None

    if cell_v_cols:
        cell_v_df = df[cell_v_cols].apply(pd.to_numeric, errors="coerce")

        col_mins = cell_v_df.min(skipna=True).dropna()
        if not col_mins.empty:
            min_col = col_mins.idxmin()
            min_cell_voltage_v = float(col_mins[min_col])
            min_cell_voltage_id = min_col.removeprefix("can_")

        col_maxes = cell_v_df.max(skipna=True).dropna()
        if not col_maxes.empty:
            max_col = col_maxes.idxmax()
            max_cell_voltage_v = float(col_maxes[max_col])
            max_cell_voltage_id = max_col.removeprefix("can_")

        # Initial imbalance: median voltage per cell over the first window, then max - min
        start_mask = ts <= _START_WINDOW_S
        start_cell = cell_v_df[start_mask]
        cell_medians = start_cell.median(skipna=True).dropna()
        if len(cell_medians) > 1:
            initial_cell_imbalance_mv = float((cell_medians.max() - cell_medians.min()) * 1000)

    return DatasetSummary(
        charged_capacity_ah=charged_capacity_ah,
        discharged_capacity_ah=discharged_capacity_ah,
        charged_energy_kwh=charged_energy_kwh,
        discharged_energy_kwh=discharged_energy_kwh,
        avg_temp_start_c=avg_temp_start_c,
        avg_max_temp_c=avg_max_temp_c,
        min_cell_voltage_v=min_cell_voltage_v,
        min_cell_voltage_id=min_cell_voltage_id,
        max_cell_voltage_v=max_cell_voltage_v,
        max_cell_voltage_id=max_cell_voltage_id,
        initial_cell_imbalance_mv=initial_cell_imbalance_mv,
    )
