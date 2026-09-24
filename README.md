# CAN log visualizer

Local Streamlit app for exploring decoded CAN signals from NDJSON logs or wide-format CSV exports.

## Run

1. Create and activate a Python 3.12 virtual environment in `.venv`.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Start the app with `.venv\Scripts\python.exe -m streamlit run app.py`.

Run the small test suite with `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Use

1. Put a `.ndjson` or `.csv` file in `data`.
2. Choose the file and one or more numeric signals.
3. Use the sidebar to limit rows, choose a time window, and reduce plot points if needed.
4. Hover or zoom in the Plotly chart; choose an export mode to download selected data.

The time axis is seconds relative to the earliest valid timestamp in the loaded rows. Invalid timestamps are skipped. Samples with out-of-order timestamps are plotted in time order; equal timestamps retain their file order.

## Inputs and limits

- NDJSON: one JSON object per line with a numeric `timestamp` and decoded values in `parsed`.
- CSV: wide-format columns with a time column named `timestamp`, `timestamp_s`, `nhr_monotonic_s`, or `nhr_timestamp_utc`. Numeric signal columns are discovered automatically; `_age_s` and `_status` columns are excluded.
- Large CSV files are read fully by default for the summary and plot index. The row limit reduces plot loading, but the summary still reads the whole file. Large files can take time and memory.

## Dataset summary

The summary covers the complete CSV recording, regardless of the chart's row limit or time window. It reads only the columns needed for its calculations. NDJSON recordings do not have a dataset summary.

- Charge is positive and discharge is negative for the measured NHR current and power. Charged/discharged capacity (Ah) and energy (kWh) integrate these signals over time, with an exact split at zero crossings. NHR cumulative counters are not used, so their resets cannot change the totals. Net values are charged minus discharged.
- Missing/nonfinite samples and timestamp gaps over 2 s are not bridged. The data-quality line reports the integrated duration and coverage for current and power. Duplicate timestamps retain the last row. Metrics without valid intervals show `N/A`.
- Battery temperature uses `can_batteryTempAvg`; minimum and maximum cell temperatures use `can_minCellTemp` and `can_maxCellTemp`. Their difference is the cell temperature spread. Time averages are weighted by valid elapsed time.
- Cell voltage extrema and their IDs come from `can_CELL_V_*`. The instantaneous cell spread requires every discovered cell-voltage channel to be valid at that timestamp. The summary reports the number of complete snapshots.
- For CAN values, a present `_status` must be `fresh` and a present `_age_s` must be between 0 and 1 s. Nonpositive cell voltages are ignored.
- Start and end values are medians over the first and last 5 s of the recording. If no valid sample falls inside a window, that value shows `N/A`.
- Enter the actual number of cells in series (`Ns`) in the sidebar to display `nhr_voltage_v / Ns` at the start and end. This is an estimate from NHR terminal voltage, including load effects, rather than an open-circuit cell measurement.

`data`, `csv_extract`, and `.venv` are excluded from Git. Keep source recordings and exported data separately from code history.
