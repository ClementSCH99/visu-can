from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.data_loader import PlotDataset, PlotTrace, SignalSummary, SignalIndex, DatasetSummary, build_signal_index, load_selected_signals, compute_dataset_summary


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SCAN_LIMIT = 50000
DEFAULT_MAX_POINTS = 5000
HOVER_MODE_OPTIONS = {
    "Unified X": "x unified",
    "X": "x",
    "Closest": "closest",
    "Unified Y": "y unified",
    "Y": "y",
    "Disabled": False,
}
EXPORT_MODE_OPTIONS = {
    "Sparse timestamps": "sparse",
    "Common timestamps with hold-last-value": "forward_fill",
}
EXPORT_RESAMPLE_OPTIONS = {
    "Use plotted timestamps": None,
    "10 ms": 0.01,
    "100 ms": 0.1,
    "1000 ms": 1.0,
}
PLOT_RESAMPLE_OPTIONS = {
    "10 ms": 0.01,
    "100 ms": 0.1,
    "1000 ms": 1.0,
}


def main() -> None:
    st.set_page_config(page_title="CAN Visualizer", layout="wide")
    st.title("CAN Visualizer")
    st.caption("Plot decoded numeric signals from NDJSON CAN logs or wide-format CSV exports.")

    source_file = select_input_file()
    if source_file is None:
        st.info("Add an NDJSON or CSV file to the data folder to get started.")
        return

    summary = cached_compute_dataset_summary(str(source_file), get_file_signature(source_file))
    show_dataset_summary(summary)

    st.sidebar.header("Controls")
    scan_limit_enabled = st.sidebar.checkbox("Limit signal discovery scan", value=True)
    scan_limit = st.sidebar.number_input(
        "Discovery rows",
        min_value=1000,
        max_value=500000,
        value=DEFAULT_SCAN_LIMIT,
        step=1000,
        disabled=not scan_limit_enabled,
        help="Only the discovery step is limited. Plot loading still reads the selected rows below.",
    )
    row_limit_enabled = st.sidebar.checkbox("Limit plotted rows", value=False)
    row_limit = st.sidebar.number_input(
        "Rows to plot",
        min_value=1000,
        max_value=5000000,
        value=200000,
        step=1000,
        disabled=not row_limit_enabled,
    )
    st.sidebar.subheader("Time window")
    start_time_text = st.sidebar.text_input(
        "Start timestamp (s)",
        value="",
        help="Leave empty to start from the beginning of the file.",
    )
    end_time_text = st.sidebar.text_input(
        "End timestamp (s)",
        value="",
        help="Leave empty to read until the end of the file.",
    )
    sampling_mode = st.sidebar.selectbox(
        "Sampling mode",
        options=["Max points", "Every Nth point", "Fixed grid"],
        index=0,
    )
    sample_every = st.sidebar.number_input(
        "Keep one point every N samples",
        min_value=1,
        max_value=1000,
        value=1,
        step=1,
        disabled=sampling_mode != "Every Nth point",
        help="Use this to apply an explicit sampling stride before Plotly rendering.",
    )
    max_points = st.sidebar.number_input(
        "Max points per signal",
        min_value=500,
        max_value=50000,
        value=DEFAULT_MAX_POINTS,
        step=500,
        disabled=sampling_mode != "Max points",
        help="Long traces are uniformly downsampled before plotting.",
    )
    plot_resample_label = st.sidebar.selectbox(
        "Fixed grid step",
        options=list(PLOT_RESAMPLE_OPTIONS.keys()),
        index=1,
        disabled=sampling_mode != "Fixed grid",
        help="Build a common timestamp axis and hold each signal value until a new sample is received.",
    )
    hover_mode_label = st.sidebar.selectbox(
        "Hover mode",
        options=list(HOVER_MODE_OPTIONS.keys()),
        index=0,
        help="Change how hover behaves, or disable it entirely.",
    )

    start_time = parse_optional_float(start_time_text, "Start timestamp")
    end_time = parse_optional_float(end_time_text, "End timestamp")
    if start_time is None and start_time_text.strip():
        return
    if end_time is None and end_time_text.strip():
        return
    if start_time is not None and end_time is not None and start_time > end_time:
        st.sidebar.error("Start timestamp must be lower than or equal to end timestamp.")
        return

    file_signature = get_file_signature(source_file)

    summaries = cached_discover_numeric_signals(
        str(source_file),
        int(scan_limit) if scan_limit_enabled else None,
        file_signature,
    )
    if not summaries:
        st.warning("No numeric decoded signals were found in the parsed payloads.")
        return

    available_signal_names = [summary.name for summary in summaries]
    available_signal_set = set(available_signal_names)

    # Preserve signal selection when switching files
    file_key = str(source_file)
    if st.session_state.get("_signals_file") != file_key:
        prev_selection = st.session_state.get("selected_signals", [])
        preserved = [s for s in prev_selection if s in available_signal_set]
        st.session_state["selected_signals"] = preserved if preserved else default_signal_selection(summaries)
        st.session_state["_signals_file"] = file_key

    selected_signals = st.multiselect(
        "Signals to plot",
        options=available_signal_names,
        key="selected_signals",
        help="Only numeric values from the parsed object are included.",
    )

    show_signal_summary(summaries)

    if not selected_signals:
        st.info("Select at least one signal to display the chart.")
        return

    axis_assignments = select_axis_assignments(selected_signals)

    current_row_limit = int(row_limit) if row_limit_enabled else None

    export_sample_every = int(sample_every) if sampling_mode == "Every Nth point" else 1
    plot_requires_full_samples = sampling_mode == "Fixed grid"

    plot_source_dataset = cached_load_selected_signals(
        str(source_file),
        tuple(selected_signals),
        current_row_limit,
        0 if plot_requires_full_samples else int(max_points) if sampling_mode == "Max points" else 0,
        start_time,
        end_time,
        export_sample_every,
        file_signature,
    )

    dataset = (
        build_plot_dataset_on_fixed_grid(plot_source_dataset, PLOT_RESAMPLE_OPTIONS[plot_resample_label])
        if sampling_mode == "Fixed grid"
        else plot_source_dataset
    )

    if not dataset.traces:
        show_empty_selection_message(
            str(source_file),
            selected_signals,
            current_row_limit,
            start_time,
            end_time,
            file_signature,
        )
        return

    st.plotly_chart(
        build_figure(dataset, HOVER_MODE_OPTIONS[hover_mode_label], axis_assignments),
        use_container_width=True,
    )
    export_mode_label = st.selectbox(
        "CSV export mode",
        options=list(EXPORT_MODE_OPTIONS.keys()),
        index=0,
        help="Sparse keeps only exact sample timestamps. Common timestamps forward-fills each signal until a new value is received.",
    )
    export_resample_label = st.selectbox(
        "CSV common timestamp grid",
        options=list(EXPORT_RESAMPLE_OPTIONS.keys()),
        index=0,
        disabled=EXPORT_MODE_OPTIONS[export_mode_label] != "forward_fill",
        help="When using common timestamps, choose whether to reuse plotted timestamps or generate a fixed resampling grid.",
    )
    export_dataset = cached_load_selected_signals(
        str(source_file),
        tuple(selected_signals),
        current_row_limit,
        0,
        start_time,
        end_time,
        export_sample_every,
        file_signature,
    )
    st.download_button(
        "Download plotted data as CSV",
        data=build_export_csv(
            export_dataset,
            EXPORT_MODE_OPTIONS[export_mode_label],
            EXPORT_RESAMPLE_OPTIONS[export_resample_label],
        ),
        file_name=f"{source_file.stem}_plot_export.csv",
        mime="text/csv",
        help="Exports the selected signals and window from the filtered source samples. `Max points` is treated as a plot-only simplification and is not used for CSV timing.",
    )
    if dataset.first_timestamp is not None and dataset.last_timestamp is not None:
        st.caption(
            f"Loaded timestamp range: {dataset.first_timestamp:.3f}s to {dataset.last_timestamp:.3f}s. "
            "Use those values in the sidebar to reload a smaller region."
        )
    st.caption(
        f"Processed {dataset.processed_rows:,} rows and found selected signals in {dataset.matched_rows:,} rows."
    )


def select_input_file() -> Path | None:
    sample_files = sorted(
        [*DATA_DIR.glob("*.ndjson"), *DATA_DIR.glob("*.csv")],
        key=lambda p: p.name,
    )
    if not sample_files:
        return None

    selected_name = st.selectbox(
        "Input file",
        options=[path.name for path in sample_files],
        index=0,
    )
    return DATA_DIR / selected_name


@st.cache_data(show_spinner=False)
def cached_discover_numeric_signals(
    file_path: str,
    scan_limit: int | None,
    file_signature: tuple[str, int, int],
) -> list[SignalSummary]:
    signal_index = cached_signal_index(file_path, scan_limit, file_signature)
    return [
        SignalSummary(name=name, samples=signal_index.signal_counts[name])
        for name in sorted(
            signal_index.signal_counts,
            key=lambda item: (-signal_index.signal_counts[item], item.lower()),
        )
    ]


@st.cache_data(show_spinner=True)
def cached_load_selected_signals(
    file_path: str,
    selected_signals: tuple[str, ...],
    row_limit: int | None,
    max_points: int,
    start_time: float | None,
    end_time: float | None,
    sample_every: int,
    file_signature: tuple[str, int, int],
):
    signal_index = cached_signal_index(file_path, row_limit, file_signature)
    return load_selected_signals(
        file_path=file_path,
        selected_signals=list(selected_signals),
        row_limit=row_limit,
        max_points_per_signal=max_points,
        start_time=start_time,
        end_time=end_time,
        sample_every=sample_every,
        signal_index=signal_index,
    )


@st.cache_resource(show_spinner=True)
def cached_signal_index(
    file_path: str,
    row_limit: int | None,
    file_signature: tuple[str, int, int],
) -> SignalIndex:
    del file_signature
    return build_signal_index(file_path, row_limit=row_limit)


@st.cache_data(show_spinner=False)
def cached_compute_dataset_summary(
    file_path: str,
    file_signature: tuple[str, int, int],
) -> DatasetSummary | None:
    del file_signature
    return compute_dataset_summary(file_path)


def get_file_signature(file_path: Path) -> tuple[str, int, int]:
    stats = file_path.stat()
    return (str(file_path), stats.st_mtime_ns, stats.st_size)


def show_empty_selection_message(
    file_path: str,
    selected_signals: list[str],
    row_limit: int | None,
    start_time: float | None,
    end_time: float | None,
    file_signature: tuple[str, int, int],
) -> None:
    limited_index = cached_signal_index(file_path, row_limit, file_signature)
    limited_range = get_selected_signal_range(limited_index, selected_signals)
    if limited_range is not None and (start_time is not None or end_time is not None):
        st.warning(
            "The selected signals exist in the loaded rows, but the current time window excludes them. "
            f"Available range in the current load: {limited_range[0]:.3f}s to {limited_range[1]:.3f}s."
        )
        return

    if row_limit is not None:
        full_index = cached_signal_index(file_path, None, file_signature)
        full_range = get_selected_signal_range(full_index, selected_signals)
        if full_range is not None:
            st.warning(
                "The selected signals exist in the file, but not in the currently loaded row range. "
                f"Increase `Rows to plot` or disable the row limit. Available file range: {full_range[0]:.3f}s to {full_range[1]:.3f}s."
            )
            return

    st.warning("None of the selected signals had numeric samples in the loaded rows.")


def get_selected_signal_range(signal_index: SignalIndex, selected_signals: list[str]) -> tuple[float, float] | None:
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    for signal_name in selected_signals:
        timestamps = signal_index.timestamps_by_signal.get(signal_name)
        row_ids = signal_index.row_ids_by_signal.get(signal_name)
        if not timestamps or not row_ids:
            continue
        if first_timestamp is None or timestamps[0] < first_timestamp:
            first_timestamp = timestamps[0]
        if last_timestamp is None or timestamps[-1] > last_timestamp:
            last_timestamp = timestamps[-1]

    if first_timestamp is None or last_timestamp is None:
        return None
    return (first_timestamp, last_timestamp)


def default_signal_selection(summaries: list[SignalSummary]) -> list[str]:
    preferred = ["batteryCurrent", "batteryVoltage", "SOC"]
    available = {summary.name for summary in summaries}
    defaults = [name for name in preferred if name in available]
    if defaults:
        return defaults
    return [summary.name for summary in summaries[:3]]


def show_signal_summary(summaries: list[SignalSummary]) -> None:
    with st.expander("Available numeric signals", expanded=False):
        st.dataframe(
            {
                "signal": [summary.name for summary in summaries],
                "samples": [summary.samples for summary in summaries],
            },
            use_container_width=True,
            hide_index=True,
        )


def _fmt(value: float | None, decimals: int = 3, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{' ' + unit if unit else ''}"


def show_dataset_summary(summary: DatasetSummary | None) -> None:
    if summary is None:
        return
    with st.expander("Dataset summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Capacity & Energy**")
            st.table(
                {
                    "Metric": [
                        "Charged capacity",
                        "Discharged capacity",
                        "Charged energy",
                        "Discharged energy",
                    ],
                    "Value": [
                        _fmt(summary.charged_capacity_ah, 3, "Ah"),
                        _fmt(summary.discharged_capacity_ah, 3, "Ah"),
                        _fmt(summary.charged_energy_kwh, 4, "kWh"),
                        _fmt(summary.discharged_energy_kwh, 4, "kWh"),
                    ],
                }
            )
        with col2:
            st.markdown("**Temperature**")
            st.table(
                {
                    "Metric": [
                        "Avg temp at start",
                        "Avg max temp",
                    ],
                    "Value": [
                        _fmt(summary.avg_temp_start_c, 1, "°C"),
                        _fmt(summary.avg_max_temp_c, 1, "°C"),
                    ],
                }
            )
            st.markdown("**Cell Voltages**")
            st.table(
                {
                    "Metric": [
                        "Min cell voltage",
                        "Max cell voltage",
                        "Initial imbalance",
                    ],
                    "Value": [
                        _fmt(summary.min_cell_voltage_v, 4, "V")
                        + (f"  ({summary.min_cell_voltage_id})" if summary.min_cell_voltage_id else ""),
                        _fmt(summary.max_cell_voltage_v, 4, "V")
                        + (f"  ({summary.max_cell_voltage_id})" if summary.max_cell_voltage_id else ""),
                        _fmt(summary.initial_cell_imbalance_mv, 1, "mV"),
                    ],
                }
            )


def select_axis_assignments(selected_signals: list[str]) -> dict[str, str]:
    with st.expander("Axis assignment", expanded=False):
        st.caption("Choose whether each signal is drawn on the left or right y-axis.")
        assignments: dict[str, str] = {}
        for signal_name in selected_signals:
            assignments[signal_name] = st.selectbox(
                f"Axis for {signal_name}",
                options=["Left", "Right"],
                index=0,
                key=f"axis-{signal_name}",
            )
        return assignments


def parse_optional_float(raw_value: str, label: str) -> float | None:
    stripped_value = raw_value.strip()
    if not stripped_value:
        return None
    try:
        return float(stripped_value)
    except ValueError:
        st.sidebar.error(f"{label} must be a valid number.")
        return None


def build_export_csv(dataset, export_mode: str, resample_step_s: float | None = None) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    signal_names = [trace.name for trace in dataset.traces]
    writer.writerow(["timestamp_s", *signal_names])

    aligned_traces = {
        trace.name: align_trace_points(trace.x, trace.y)
        for trace in dataset.traces
    }

    if export_mode == "forward_fill":
        write_forward_filled_rows(writer, signal_names, aligned_traces, resample_step_s)
    else:
        write_sparse_rows(writer, signal_names, aligned_traces)

    return buffer.getvalue()


def write_sparse_rows(
    writer: Any,
    signal_names: list[str],
    aligned_traces: dict[str, tuple[list[float], list[float]]],
) -> None:
    rows_by_timestamp: dict[float, dict[str, float]] = {}
    for signal_name, (timestamps, values) in aligned_traces.items():
        for timestamp, value in zip(timestamps, values):
            if timestamp not in rows_by_timestamp:
                rows_by_timestamp[timestamp] = {}
            rows_by_timestamp[timestamp][signal_name] = value

    for timestamp in sorted(rows_by_timestamp):
        signal_values = rows_by_timestamp[timestamp]
        writer.writerow([timestamp, *[signal_values.get(signal_name, "") for signal_name in signal_names]])


def write_forward_filled_rows(
    writer: Any,
    signal_names: list[str],
    aligned_traces: dict[str, tuple[list[float], list[float]]],
    resample_step_s: float | None,
) -> None:
    timestamps = build_forward_fill_timestamps(aligned_traces, resample_step_s)
    if not timestamps:
        return

    signal_positions = {signal_name: 0 for signal_name in signal_names}
    last_values: dict[str, float] = {}

    for timestamp in timestamps:
        row: list[float | str] = [timestamp]
        for signal_name in signal_names:
            trace_timestamps, trace_values = aligned_traces[signal_name]
            position = signal_positions[signal_name]
            while position < len(trace_timestamps) and trace_timestamps[position] <= timestamp:
                last_values[signal_name] = trace_values[position]
                position += 1
            signal_positions[signal_name] = position
            row.append(last_values.get(signal_name, ""))
        writer.writerow(row)


def build_forward_fill_timestamps(
    aligned_traces: dict[str, tuple[list[float], list[float]]],
    resample_step_s: float | None,
) -> list[float]:
    if resample_step_s is None:
        return sorted(
            {
                timestamp
                for trace_timestamps, _ in aligned_traces.values()
                for timestamp in trace_timestamps
            }
        )

    ranges = [
        (trace_timestamps[0], trace_timestamps[-1])
        for trace_timestamps, _ in aligned_traces.values()
        if trace_timestamps
    ]
    if not ranges:
        return []

    start_timestamp = min(start for start, _ in ranges)
    end_timestamp = max(end for _, end in ranges)
    timestamps: list[float] = []
    current_timestamp = start_timestamp
    while current_timestamp <= end_timestamp:
        timestamps.append(round(current_timestamp, 9))
        current_timestamp += resample_step_s
    if timestamps[-1] < end_timestamp:
        timestamps.append(end_timestamp)
    return timestamps


def build_plot_dataset_on_fixed_grid(dataset: PlotDataset, resample_step_s: float) -> PlotDataset:
    aligned_traces = {
        trace.name: align_trace_points(trace.x, trace.y)
        for trace in dataset.traces
    }
    timestamps = build_forward_fill_timestamps(aligned_traces, resample_step_s)
    if not timestamps:
        return dataset

    signal_positions = {trace.name: 0 for trace in dataset.traces}
    last_values: dict[str, float] = {}
    resampled_values: dict[str, list[float]] = {trace.name: [] for trace in dataset.traces}
    resampled_timestamps: dict[str, list[float]] = {trace.name: [] for trace in dataset.traces}

    for timestamp in timestamps:
        for trace in dataset.traces:
            trace_timestamps, trace_values = aligned_traces[trace.name]
            position = signal_positions[trace.name]
            while position < len(trace_timestamps) and trace_timestamps[position] <= timestamp:
                last_values[trace.name] = trace_values[position]
                position += 1
            signal_positions[trace.name] = position
            if trace.name in last_values:
                resampled_timestamps[trace.name].append(timestamp)
                resampled_values[trace.name].append(last_values[trace.name])

    return PlotDataset(
        traces=[
            PlotTrace(
                name=trace.name,
                x=resampled_timestamps[trace.name],
                y=resampled_values[trace.name],
            )
            for trace in dataset.traces
            if resampled_timestamps[trace.name]
        ],
        processed_rows=dataset.processed_rows,
        matched_rows=dataset.matched_rows,
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
    )


def align_trace_points(timestamps: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    point_count = min(len(timestamps), len(values))
    return (timestamps[:point_count], values[:point_count])


def build_figure(dataset, hover_mode: str | bool, axis_assignments: dict[str, str]):
    figure = go.Figure()
    uses_secondary_axis = False
    for trace in dataset.traces:
        is_right_axis = axis_assignments.get(trace.name, "Left") == "Right"
        if is_right_axis:
            uses_secondary_axis = True
        timestamps, values = align_trace_points(trace.x, trace.y)
        figure.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=trace.name,
                yaxis="y2" if is_right_axis else "y",
            )
        )

    figure.update_layout(
        xaxis_title="Timestamp (s)",
        yaxis_title="Left axis",
        hovermode=hover_mode,
        legend_title_text="Signals",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    if uses_secondary_axis:
        figure.update_layout(
            yaxis2=dict(
                title="Right axis",
                overlaying="y",
                side="right",
            )
        )
    return figure


if __name__ == "__main__":
    main()