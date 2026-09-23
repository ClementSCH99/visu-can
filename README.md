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

`data`, `csv_extract`, and `.venv` are excluded from Git. Keep source recordings and exported data separately from code history.
