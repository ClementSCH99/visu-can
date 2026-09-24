from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


START_END_WINDOW_S = 5.0
MAX_INTEGRATION_GAP_S = 2.0
MAX_CAN_AGE_S = 1.0
_TIME_COLUMNS = ("timestamp", "timestamp_s", "nhr_monotonic_s", "nhr_timestamp_utc")
_SCALARS = (
    "nhr_current_a", "nhr_power_w", "nhr_voltage_v",
    "can_batteryTempAvg", "can_minCellTemp", "can_maxCellTemp",
)


@dataclass(frozen=True)
class IntegratedQuantity:
    charged: float | None
    discharged: float | None
    net: float | None
    covered_s: float
    skipped_intervals: int


@dataclass(frozen=True)
class SeriesStats:
    start: float | None
    end: float | None
    minimum: float | None
    maximum: float | None
    average: float | None


@dataclass(frozen=True)
class CellExtreme:
    value_v: float | None
    cell_id: str | None
    time_s: float | None


@dataclass(frozen=True)
class DatasetSummary:
    duration_s: float
    capacity_ah: IntegratedQuantity
    energy_kwh: IntegratedQuantity
    battery_temp_c: SeriesStats
    min_cell_temp_c: SeriesStats
    max_cell_temp_c: SeriesStats
    cell_temp_delta_c: SeriesStats
    min_cell_voltage: CellExtreme
    max_cell_voltage: CellExtreme
    cell_voltage_delta_mv: SeriesStats
    equivalent_cell_voltage_v: SeriesStats
    series_cells: int | None
    complete_cell_snapshots: int
    total_rows: int
    warnings: tuple[str, ...]


def _empty_stats() -> SeriesStats:
    return SeriesStats(None, None, None, None, None)


def _empty_extreme() -> CellExtreme:
    return CellExtreme(None, None, None)


def _values(df: pd.DataFrame, name: str, *, can: bool = False, positive: bool = False) -> np.ndarray:
    if name not in df:
        return np.full(len(df), np.nan)
    values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values)
    if positive:
        valid &= values > 0
    if can:
        status = f"{name}_status"
        age = f"{name}_age_s"
        if status in df:
            valid &= df[status].astype("string").str.lower().eq("fresh").fillna(False).to_numpy(dtype=bool)
        if age in df:
            ages = pd.to_numeric(df[age], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            valid &= np.isfinite(ages) & (ages >= 0) & (ages <= MAX_CAN_AGE_S)
    return np.where(valid, values, np.nan)


def _valid_pairs(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(t)
    valid = (dt > 0) & (dt <= MAX_INTEGRATION_GAP_S)
    valid &= np.isfinite(x[:-1]) & np.isfinite(x[1:])
    return dt, valid


def _integrate(t: np.ndarray, x: np.ndarray, divisor: float) -> IntegratedQuantity:
    if len(t) < 2:
        return IntegratedQuantity(None, None, None, 0.0, 0)
    dt, valid = _valid_pairs(t, x)
    skipped = int((~valid).sum())
    if not valid.any():
        return IntegratedQuantity(None, None, None, 0.0, skipped)
    d, a, b = dt[valid], x[:-1][valid], x[1:][valid]
    charge = d * (np.maximum(a, 0) + np.maximum(b, 0)) / 2
    discharge = d * (np.maximum(-a, 0) + np.maximum(-b, 0)) / 2
    crossing = (a < 0) & (b > 0) | (a > 0) & (b < 0)
    if crossing.any():
        fraction = np.abs(a[crossing]) / (np.abs(a[crossing]) + np.abs(b[crossing]))
        positive_first = a[crossing] > 0
        charge[crossing] = np.where(
            positive_first,
            d[crossing] * np.abs(a[crossing]) * fraction / 2,
            d[crossing] * np.abs(b[crossing]) * (1 - fraction) / 2,
        )
        discharge[crossing] = np.where(
            positive_first,
            d[crossing] * np.abs(b[crossing]) * (1 - fraction) / 2,
            d[crossing] * np.abs(a[crossing]) * fraction / 2,
        )
    charged = float(charge.sum() / divisor)
    discharged = float(discharge.sum() / divisor)
    return IntegratedQuantity(charged, discharged, charged - discharged, float(d.sum()), skipped)


def _stats(t: np.ndarray, x: np.ndarray) -> SeriesStats:
    valid = np.isfinite(x)
    if not valid.any():
        return _empty_stats()
    start = x[valid & (t <= START_END_WINDOW_S)]
    end = x[valid & (t >= t[-1] - START_END_WINDOW_S)]
    dt, pairs = _valid_pairs(t, x) if len(t) > 1 else (np.array([]), np.array([], dtype=bool))
    average = (
        float(np.sum((x[:-1][pairs] + x[1:][pairs]) * dt[pairs] / 2) / dt[pairs].sum())
        if pairs.any() else None
    )
    return SeriesStats(
        float(np.median(start)) if len(start) else None,
        float(np.median(end)) if len(end) else None,
        float(np.min(x[valid])),
        float(np.max(x[valid])),
        average,
    )


def _extreme(t: np.ndarray, cells: np.ndarray, names: list[str], *, maximum: bool) -> CellExtreme:
    valid = np.isfinite(cells)
    if not valid.any():
        return _empty_extreme()
    candidate = np.where(valid, cells, -np.inf if maximum else np.inf)
    row, column = np.unravel_index(np.argmax(candidate) if maximum else np.argmin(candidate), cells.shape)
    return CellExtreme(float(cells[row, column]), names[column].removeprefix("can_"), float(t[row]))


def compute_dataset_summary(file_path: str | Path, series_cells: int | None = None) -> DatasetSummary | None:
    path = Path(file_path)
    if path.suffix.lower() != ".csv":
        return None
    if series_cells is not None and series_cells < 1:
        raise ValueError("Cells in series must be a positive integer")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    time_col = next((name for name in _TIME_COLUMNS if name in columns), None)
    if time_col is None:
        return None
    cell_names = sorted(name for name in columns if name.startswith("can_CELL_V_") and not name.endswith(("_status", "_age_s")))
    names = [name for name in _SCALARS if name in columns] + cell_names
    wanted = {time_col}
    for name in names:
        wanted.add(name)
        if name.startswith("can_"):
            wanted.update({f"{name}_status", f"{name}_age_s"} & set(columns))
    df = pd.read_csv(path, usecols=lambda name: name in wanted)
    if df.empty:
        return None
    if time_col == "nhr_timestamp_utc":
        parsed = pd.to_datetime(df[time_col], utc=True, errors="coerce", format="ISO8601")
        timestamps = (parsed.dt.as_unit("ns").astype("int64") / 1e9).where(parsed.notna())
    else:
        timestamps = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(_t=timestamps)
    df = df[np.isfinite(df["_t"])].sort_values("_t", kind="stable")
    if df.empty:
        return None
    duplicates = int(df["_t"].duplicated().sum())
    df = df.drop_duplicates("_t", keep="last").reset_index(drop=True)
    t = df["_t"].to_numpy(dtype=float, copy=True)
    t -= t[0]
    duration = float(t[-1])
    warnings: list[str] = []
    if duplicates:
        warnings.append(f"{duplicates} duplicate timestamps: last row retained.")
    if len(t) > 1 and (np.diff(t) > MAX_INTEGRATION_GAP_S).any():
        warnings.append("Time gaps over 2 s were excluded from integration and time averages.")

    capacity = _integrate(t, _values(df, "nhr_current_a"), 3600)
    energy = _integrate(t, _values(df, "nhr_power_w"), 3_600_000)
    for label, result in (("Current", capacity), ("Power", energy)):
        if result.charged is None:
            warnings.append(f"{label}: no valid interval for integration.")
        elif duration and result.covered_s / duration < 0.999:
            warnings.append(f"{label}: partial integration coverage ({result.covered_s / duration:.1%}).")

    battery_temp = _values(df, "can_batteryTempAvg", can=True)
    min_temp = _values(df, "can_minCellTemp", can=True)
    max_temp = _values(df, "can_maxCellTemp", can=True)
    temp_delta = np.where(np.isfinite(min_temp) & np.isfinite(max_temp) & (max_temp >= min_temp), max_temp - min_temp, np.nan)
    for label, values in (
        ("CAN battery average temperature", battery_temp),
        ("CAN minimum cell temperature", min_temp),
        ("CAN maximum cell temperature", max_temp),
        ("CAN cell temperature spread", temp_delta),
    ):
        if not np.isfinite(values).any():
            warnings.append(f"{label} unavailable or stale.")

    cells = np.column_stack([_values(df, name, can=True, positive=True) for name in cell_names]) if cell_names else np.empty((len(df), 0))
    complete = np.isfinite(cells).all(axis=1) if cell_names else np.zeros(len(df), dtype=bool)
    delta_v = np.full(len(df), np.nan)
    if complete.any():
        delta_v[complete] = np.ptp(cells[complete], axis=1) * 1000
    else:
        warnings.append("No complete, fresh cell-voltage snapshot is available.")
    if cell_names and int(complete.sum()) < len(df):
        warnings.append(f"Complete cell-voltage snapshots: {int(complete.sum())}/{len(df)} rows.")

    voltage = _values(df, "nhr_voltage_v", positive=True)
    equivalent = _stats(t, voltage / series_cells) if series_cells is not None else _empty_stats()
    if series_cells is None:
        warnings.append("Enter the number of cells in series to display Vpack/Ns.")
    elif equivalent.start is None or equivalent.end is None:
        warnings.append("NHR terminal voltage missing from the start or end window.")

    return DatasetSummary(
        duration, capacity, energy, _stats(t, battery_temp), _stats(t, min_temp),
        _stats(t, max_temp), _stats(t, temp_delta),
        _extreme(t, cells, cell_names, maximum=False) if cell_names else _empty_extreme(),
        _extreme(t, cells, cell_names, maximum=True) if cell_names else _empty_extreme(),
        _stats(t, delta_v), equivalent, series_cells, int(complete.sum()), len(df), tuple(warnings),
    )
