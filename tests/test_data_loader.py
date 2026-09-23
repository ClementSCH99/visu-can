import unittest
from pathlib import Path
from uuid import uuid4

from src.data_loader import build_sample_indices, build_signal_index, load_selected_signals


class DataLoaderTests(unittest.TestCase):
    def fixture(self, name, content):
        path = Path.cwd() / f"_test_fixture_{uuid4().hex}_{name}"
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
