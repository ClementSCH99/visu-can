import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.summary import compute_dataset_summary


class SummaryTests(unittest.TestCase):
    def fixture(self, rows):
        root = Path.cwd() / ".codex_tmp"
        root.mkdir(exist_ok=True)
        path = root / f"summary_{uuid4().hex}.csv"
        self.addCleanup(path.unlink, missing_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_signed_integration_ignores_reset_counters_and_splits_zero_crossing(self):
        path = self.fixture({
            "timestamp": [0, 1, 2, 3],
            "nhr_current_a": [2, 2, -2, -2],
            "nhr_power_w": [200, 200, -200, -200],
            "nhr_capacity_charge_ah": [10, 11, 0, 1],
            "nhr_energy_charge_kwh": [10, 11, 0, 1],
        })
        summary = compute_dataset_summary(path)
        self.assertAlmostEqual(summary.capacity_ah.charged, 2.5 / 3600)
        self.assertAlmostEqual(summary.capacity_ah.discharged, 2.5 / 3600)
        self.assertAlmostEqual(summary.energy_kwh.charged, 250 / 3_600_000)
        self.assertAlmostEqual(summary.energy_kwh.discharged, 250 / 3_600_000)
        self.assertAlmostEqual(summary.capacity_ah.net, 0)

    def test_gap_and_invalid_sample_are_not_integrated(self):
        path = self.fixture({
            "timestamp": [0, 1, 2, 5, 6, 7],
            "nhr_current_a": [1, 1, None, 100, 100, 100],
            "nhr_power_w": [10, 10, None, 1000, 1000, 1000],
        })
        summary = compute_dataset_summary(path)
        self.assertAlmostEqual(summary.capacity_ah.charged, 201 / 3600)
        self.assertAlmostEqual(summary.capacity_ah.covered_s, 3)
        self.assertEqual(summary.capacity_ah.skipped_intervals, 2)
        self.assertTrue(any("partial integration" in item for item in summary.warnings))

    def test_five_second_windows_cell_ids_and_can_freshness(self):
        timestamps = list(range(13))
        path = self.fixture({
            "timestamp": timestamps,
            "nhr_voltage_v": [100] * 6 + [120] * 7,
            "can_batteryTempAvg": [20] * 6 + [30] * 7,
            "can_batteryTempAvg_status": ["fresh"] * 13,
            "can_minCellTemp": [19] * 6 + [28] * 7,
            "can_maxCellTemp": [21] * 6 + [33] * 7,
            "can_CELL_V_0_0": [3.0] * 6 + [4.0] * 7,
            "can_CELL_V_0_0_status": ["fresh"] * 13,
            "can_CELL_V_0_1": [3.1] * 6 + [4.2] * 7,
            "can_CELL_V_0_1_status": ["fresh"] * 12 + ["stale"],
        })
        summary = compute_dataset_summary(path, series_cells=20)
        self.assertEqual(summary.battery_temp_c.start, 20)
        self.assertEqual(summary.battery_temp_c.end, 30)
        self.assertEqual(summary.min_cell_temp_c.minimum, 19)
        self.assertEqual(summary.max_cell_temp_c.maximum, 33)
        self.assertEqual(summary.cell_temp_delta_c.start, 2)
        self.assertEqual(summary.cell_temp_delta_c.end, 5)
        self.assertEqual(summary.min_cell_voltage.cell_id, "CELL_V_0_0")
        self.assertEqual(summary.max_cell_voltage.cell_id, "CELL_V_0_1")
        self.assertAlmostEqual(summary.cell_voltage_delta_mv.start, 100)
        self.assertAlmostEqual(summary.cell_voltage_delta_mv.end, 200)
        self.assertEqual(summary.complete_cell_snapshots, 12)
        self.assertEqual(summary.equivalent_cell_voltage_v.start, 5)
        self.assertEqual(summary.equivalent_cell_voltage_v.end, 6)

    def test_age_and_duplicate_timestamps_do_not_create_false_cell_spread(self):
        path = self.fixture({
            "timestamp": [0, 1, 1, 2],
            "can_CELL_V_0_0": [3.0, 3.0, 3.0, 3.0],
            "can_CELL_V_0_0_age_s": [0.1, 0.1, 0.1, 0.1],
            "can_CELL_V_0_1": [3.1, 5.0, 3.2, 9.0],
            "can_CELL_V_0_1_age_s": [0.1, 0.1, 0.1, 1.5],
        })
        summary = compute_dataset_summary(path)
        self.assertEqual(summary.total_rows, 3)
        self.assertEqual(summary.complete_cell_snapshots, 2)
        self.assertAlmostEqual(summary.cell_voltage_delta_mv.maximum, 200)
        self.assertEqual(summary.max_cell_voltage.value_v, 3.2)
        self.assertTrue(any("duplicate timestamps" in item for item in summary.warnings))


if __name__ == "__main__":
    unittest.main()
