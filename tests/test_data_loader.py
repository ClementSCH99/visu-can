import unittest
import warnings
from pathlib import Path
from uuid import uuid4

from pandas.errors import PerformanceWarning

from src.data_loader import build_sample_indices, build_signal_index, load_selected_signals


class DataLoaderTests(unittest.TestCase):
    def fixture(self, name, content):
        root = Path.cwd() / ".codex_tmp"
        root.mkdir(exist_ok=True)
        path = root / f"_test_fixture_{uuid4().hex}_{name}"
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_csv_invalid_utc_timestamp_is_skipped(self):
        path = self.fixture(
            "utc.csv",
            "nhr_timestamp_utc,signal\n"
            "invalid,1\n"
            "2026-01-01T00:00:00Z,2\n"
            "2026-01-01T00:00:01Z,3\n",
        )
        index = build_signal_index(path)
        self.assertEqual(index.processed_rows, 3)
        self.assertEqual(index.timestamps_by_signal["signal"], [0.0, 1.0])
        self.assertEqual(index.values_by_signal["signal"], [2.0, 3.0])

    def test_out_of_order_csv_is_sorted_before_time_filtering(self):
        path = self.fixture("order.csv", "timestamp,signal\n0,10\n2,20\n1,30\n")
        index = build_signal_index(path)
        dataset = load_selected_signals(path, ["signal"], start_time=1, end_time=1.5, signal_index=index)
        self.assertEqual(index.timestamps_by_signal["signal"], [0.0, 1.0, 2.0])
        self.assertEqual(dataset.traces[0].y, [30.0])

    def test_wide_csv_indexes_without_fragmentation_warning(self):
        columns = [f"signal_{i}" for i in range(120)]
        header = "timestamp," + ",".join(columns) + "\n"
        rows = [
            "2," + ",".join([str(value)] + [""] * 119) + "\n"
            for value in (20, 21)
        ]
        rows.insert(0, "invalid," + ",".join(["9"] + [""] * 119) + "\n")
        rows.append("1," + ",".join(["10"] + [""] * 119) + "\n")
        path = self.fixture("wide.csv", header + "".join(rows))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PerformanceWarning)
            index = build_signal_index(path)

        self.assertFalse(any(issubclass(item.category, PerformanceWarning) for item in caught))
        self.assertEqual(index.processed_rows, 4)
        self.assertEqual(index.timestamps_by_signal["signal_0"], [0.0, 1.0, 1.0])
        self.assertEqual(index.values_by_signal["signal_0"], [10.0, 20.0, 21.0])

    def test_out_of_order_ndjson_is_sorted_before_time_filtering(self):
        path = self.fixture(
            "order.ndjson",
            '{"timestamp":0,"parsed":{"signal":10}}\n'
            '{"timestamp":2,"parsed":{"signal":20}}\n'
            '{"timestamp":1,"parsed":{"signal":30}}\n',
        )
        dataset = load_selected_signals(path, ["signal"], start_time=1, end_time=1.5)
        self.assertEqual(dataset.traces[0].x, [1.0])
        self.assertEqual(dataset.traces[0].y, [30.0])

    def test_downsampling_keeps_endpoints_within_limit(self):
        indices = build_sample_indices(1001, sample_every=1, max_points=500)
        self.assertEqual(len(indices), 500)
        self.assertEqual((indices[0], indices[-1]), (0, 1000))
        self.assertEqual(indices, sorted(set(indices)))


if __name__ == "__main__":
    unittest.main()
