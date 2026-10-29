"""检查百分比误差定义及图中预测来源；测试数据仅在临时目录中生成。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_训练流程/05_evaluate"))
from metrics import metrics
from plot_predictions import save_prediction_figures
from report import save_unified_results


class MetricsAndFiguresTest(unittest.TestCase):
    def test_metric_values_and_zero_denominator(self):
        scores = metrics([110., 180.], [100., 200.], [100., 100.])
        self.assertEqual(scores["MAE"], 15.)
        self.assertAlmostEqual(scores["MAPE"], 10.)
        self.assertAlmostEqual(scores["RMSE"], np.sqrt(250.))
        self.assertAlmostEqual(scores["skill_vs_persistence"], .95)
        self.assertIsNone(metrics([1.], [0.], [1.])["MAPE"])

    def test_report_preserves_percentage_units_and_seed_spread(self):
        cfg = {"tasks": {"A": []}, "models": ["TimeXer"], "seeds": [42, 123]}
        rows = [{"task": "A", "model": "TimeXer", "seed": seed, "lr": .001, "MAE": 1.,
                 "MAPE": value, "RMSE": 2., "skill_vs_persistence": .5, "training_seconds": 10.}
                for seed, value in [(42, 1.), (123, 3.)]]
        rows.append({"task": "A", "model": "Persistence", "MAE": 2., "MAPE": 4.,
                     "RMSE": 3., "skill_vs_persistence": 0.})
        with tempfile.TemporaryDirectory() as folder:
            result = save_unified_results(rows, cfg, Path(folder) / "report.csv")
            self.assertEqual(next(row for row in result if row["记录类型"] == "均值")["测试MAPE(%)"], 2.)
            self.assertAlmostEqual(next(row for row in result if row["记录类型"] == "标准差")["测试MAPE(%)"], np.sqrt(2.))

    def test_figures_use_all_aligned_points_and_fixed_seed(self):
        cfg = {"models": ["TimeXer"], "seeds": [42, 123], "frequency_minutes": 30}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {"TimeXer": {}}
            time = pd.date_range("2025-01-01", periods=8, freq="30min")
            for seed in cfg["seeds"]:
                path = root / f"predictions_{seed}.csv"
                pd.DataFrame({"time": time, "observed": np.linspace(141., 142., 8),
                              "predicted": np.linspace(141., 142., 8) + .01}).to_csv(path, index=False)
                files["TimeXer"][seed] = path
            metadata = save_prediction_figures(cfg, "A", files, root / "figures")
            self.assertEqual(metadata["reference_seed"], 42)
            self.assertEqual(metadata["n_test_samples"], 8)
            self.assertEqual(len(metadata["generated_files"]), 9)
            for name in metadata["generated_files"]:
                self.assertGreater((root / "figures" / name).stat().st_size, 0)
            saved = json.loads((root / "figures/figure_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["all_seeds"], [42, 123])
            frame = pd.read_csv(files["TimeXer"][123])
            frame.loc[0, "observed"] += 1
            frame.to_csv(files["TimeXer"][123], index=False)
            with self.assertRaises(AssertionError):
                save_prediction_figures(cfg, "A", files, root / "invalid")


if __name__ == "__main__":
    unittest.main()
