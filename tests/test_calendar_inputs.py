"""检查日历原值进入模型、物理归一化和目标还原；不执行优化器更新。"""
from io import BytesIO
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_训练流程/03_模型设计"))
from adapters import build, training_loss


class CalendarInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = json.loads((ROOT / "04_实验配置/protocol.json").read_text(encoding="utf-8"))
        cls.meta = json.loads((ROOT / "01_数据集/03_划分打包/packing_audit.json").read_text(encoding="utf-8"))["tasks"]["A"]
        cls.calendar = cls.meta["time_feature_indices"]
        cls.physical = [i for i in range(cls.meta["total_channels"]) if i not in cls.calendar]
        with np.load(ROOT / "01_数据集/03_划分打包/A/train.npz") as pack:
            cls.sample = torch.tensor(pack["x"][:4])
        # 同时覆盖变化的日内位置和不变但非零的星期、月内周次。
        cls.sample[:, :, cls.calendar[0]] = torch.linspace(0.1, 0.3, cls.cfg["seq_len"])
        cls.sample[:, :, cls.calendar[1]] = 2 / 6
        cls.sample[:, :, cls.calendar[2]] = 3 / 4

    def model(self, name):
        torch.manual_seed(42)
        return build(name, self.meta, self.cfg, torch.device("cpu")).eval()

    def test_embedding_inputs_and_target_scale(self):
        for name in ("TimeXer", "iTransformer", "TimeFilter"):
            with self.subTest(model=name):
                model = self.model(name)
                net, captured = model.network, {}

                def capture_input(module, args):
                    captured["input"] = args[0].detach().clone()
                    if len(args) > 1:
                        captured["calendar"] = args[1].detach().clone()

                def capture_head(module, args, output):
                    captured["head"] = output.detach().clone()

                embedding = net.ex_embedding if name == "TimeXer" else net.enc_embedding if name == "iTransformer" else net.patch_embed
                head = net.projection if name == "iTransformer" else net.head
                hooks = [embedding.register_forward_pre_hook(capture_input), head.register_forward_hook(capture_head)]
                x = self.sample.clone()
                try:
                    with torch.inference_mode():
                        prediction = model(x)
                finally:
                    for hook in hooks:
                        hook.remove()
                torch.testing.assert_close(x, self.sample, rtol=0, atol=0)
                self.assertEqual(tuple(prediction.shape), (4, 1))
                self.assertTrue(torch.isfinite(prediction).all())

                mean = x.mean(1, keepdim=True)
                std = torch.sqrt((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5)
                normalized = (x - mean) / std
                if name == "TimeFilter":
                    actual = captured["input"].reshape(4, x.shape[2], x.shape[1]).permute(0, 2, 1)
                    torch.testing.assert_close(actual[:, :, self.calendar], x[:, :, self.calendar], rtol=0, atol=0)
                    torch.testing.assert_close(actual[:, :, self.physical], normalized[:, :, self.physical])
                    raw_target = captured["head"][:, -1, :]
                else:
                    expected_indices = self.physical[:-1] if name == "TimeXer" else self.physical
                    torch.testing.assert_close(captured["input"], normalized[:, :, expected_indices])
                    torch.testing.assert_close(captured["calendar"], x[:, :, self.calendar], rtol=0, atol=0)
                    # iTransformer时间标记追加在物理变量后，目标仍是最后一个物理变量。
                    target_token = 0 if name == "TimeXer" else len(self.physical) - 1
                    raw_target = captured["head"][:, target_token, :]
                expected_prediction = raw_target * std[:, 0, -1:] + mean[:, 0, -1:]
                torch.testing.assert_close(prediction, expected_prediction)

    def test_calendar_changes_target_and_backward_is_finite(self):
        for name in ("TimeXer", "iTransformer", "TimeFilter"):
            with self.subTest(model=name):
                model = self.model(name)
                model.train()
                prediction = model(self.sample)
                loss = training_loss(model, prediction, torch.zeros_like(prediction), self.cfg)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                if name == "TimeFilter":
                    gate_gradients = [p.grad for key, p in model.named_parameters() if "mask_moe.gate" in key]
                    self.assertTrue(gate_gradients)
                    self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in gate_gradients))
                model.eval()
                for index in self.calendar:
                    changed = self.sample.clone()
                    changed[:, :, index] += 0.1
                    with torch.inference_mode():
                        difference = (model(changed) - model(self.sample)).abs().max()
                    self.assertGreater(float(difference), 0)

    def test_save_reload_and_single_sample(self):
        for name in ("TimeXer", "iTransformer", "TimeFilter"):
            with self.subTest(model=name):
                model = self.model(name)
                buffer = BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = self.model(name)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                with torch.inference_mode():
                    for batch_size in (1, 4):
                        x = self.sample[:batch_size]
                        prediction = restored(x)
                        self.assertEqual(tuple(prediction.shape), (batch_size, 1))
                        torch.testing.assert_close(prediction, model(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
