"""验证并行调度仍按验证集选参、保持49项不重复任务，并隔离进度目录。"""
from pathlib import Path
import importlib.util
import json
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parallel_entry", ROOT / "parallel_train.py")
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


class ParallelPlanTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "04_实验配置/protocol.json").read_text(encoding="utf-8"))

    def test_no_extra_seeds_before_learning_rate_selection(self):
        jobs, selected = parallel.ready_jobs(self.cfg, lambda job: None)
        self.assertEqual(len(jobs), 21)
        self.assertFalse(selected)
        self.assertTrue(all(job["seed"] == 42 for job in jobs))

    def test_selected_learning_rate_and_all_49_unique_jobs(self):
        def completed_candidate(job):
            return {**job, "best_val_RMSE": {0.0001: .03, 0.0003: .01, 0.001: .02}[job["lr"]]}

        jobs, selected = parallel.ready_jobs(self.cfg, completed_candidate)
        self.assertEqual(len(jobs), 49)
        self.assertEqual(len({parallel.key(job) for job in jobs}), 49)
        self.assertEqual(len(selected), 7)
        self.assertTrue(all(row["lr"] == .0003 for row in selected.values()))
        for name in self.cfg["models"]:
            group = [job for job in jobs if job["model"] == name]
            self.assertEqual(len(group), 7)
            self.assertEqual({job["seed"] for job in group}, set(self.cfg["seeds"]))
            self.assertTrue(all(job["lr"] == .0003 for job in group if job["seed"] != 42))

    def test_worker_reuses_training_function_with_separate_progress(self):
        job = {"task": "A", "model": "TimeXer", "lr": .0003, "seed": 123}
        with patch.object(parallel.training, "configuration", return_value=self.cfg), \
             patch.object(parallel.training, "prepare", return_value={"audit": True}), \
             patch.object(parallel.training, "fingerprint", return_value="same-source"), \
             patch.object(parallel.training, "RUNTIME", parallel.RUNTIME), \
             patch.object(Path, "mkdir"), \
             patch.object(parallel.training, "train_job") as train:
            parallel.worker(job, "same-source")
            self.assertEqual(parallel.training.RUNTIME, parallel.RUNTIME / "parallel_workers" / parallel.key(job))
            train.assert_called_once_with(self.cfg, {"audit": True}, job, "same-source")

    def test_changed_source_is_rejected(self):
        with patch.object(parallel.training, "configuration", return_value=self.cfg), \
             patch.object(parallel.training, "prepare", return_value={}), \
             patch.object(parallel.training, "fingerprint", return_value="changed"), \
             patch.object(parallel.training, "train_job") as train:
            with self.assertRaises(ValueError):
                parallel.worker({"task": "A", "model": "HPA", "lr": .001, "seed": 123}, "original")
            train.assert_not_called()

    def test_adopt_worker_before_job_metadata_is_written(self):
        job = {"task": "A", "model": "HPA", "lr": .001, "seed": 2024}
        # /proc stat第22项为启动时钟计数；进程名中的空格不影响解析。
        process_stat = "123 (python worker) " + " ".join(["S"] + ["0"] * 18 + ["12345"])
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", side_effect=[process_stat, "cpu 1 2 3\nbtime 1000\n"]), \
             patch.object(parallel.os, "sysconf", return_value=100, create=True):
            self.assertEqual(parallel.started_at(123, job), 1123.45)


if __name__ == "__main__":
    unittest.main()
