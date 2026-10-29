"""加载本轮主方案的五个独立种子权重之一，预测标准化历史输入。

默认只预测已有验证包；也可用--input传入含x的npz，x须沿用原字段顺序和训练标准化。
这不是五模型集成；不依据测试成绩选择种子，也不重新拟合标准化参数。
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import run_improvement as experiment


def main():
    parser = argparse.ArgumentParser(description="用当前主方案预测下一步十二里河水位")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2024, 2025, 3407], default=42)
    parser.add_argument("--input", type=Path, help="含标准化x[B,12,17]的npz；缺省为原验证数据包")
    parser.add_argument("--output", type=Path, help="新预测CSV路径；不允许覆盖已有文件")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    receipt = experiment.base.read(experiment.OUT / "fit_complete.json")
    if receipt["signature"] != experiment.signature():
        raise RuntimeError("当前模型/训练适配代码与训练记录不一致")
    cfg, audit = experiment.context(args.device)
    if args.input is None:
        pack = experiment.base.load("A", "val")
    else:
        with np.load(args.input, allow_pickle=False) as values:
            pack = {name: values[name] for name in values.files}
    x = np.asarray(pack["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (12, 17) or not np.isfinite(x).all():
        raise ValueError("输入必须为有限值的标准化历史x[B,12,17]，字段顺序与原数据包相同")
    device = experiment.base.setup(cfg, args.seed)
    model = experiment.tv.build_variant(experiment.PRIMARY_VARIANT, audit["tasks"]["A"], cfg, device)
    weight = experiment.variant_weight(experiment.PRIMARY_VARIANT, args.seed)
    job = {"task": "A", "model": experiment.PRIMARY_VARIANT, "seed": args.seed,
           "lr": experiment.selected_lr(experiment.PRIMARY_VARIANT)}
    if experiment.done(job) is None:
        raise RuntimeError("该种子的训练尚未完成")
    model.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
    mean, std = experiment.base.scale("A")
    pred = experiment.base.predict(model, torch.tensor(x, device=device), cfg["batch_size"]) * std + mean
    target = args.output or experiment.OUT / f"使用预测_seed{args.seed}.csv"
    if target.exists():
        raise FileExistsError(f"预测文件已存在，请指定新的--output路径：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"predicted_water_level_m": pred})
    if "time" in pack:
        result.insert(0, "time", pack["time"])
    result.to_csv(target, index=False, encoding="utf-8-sig")
    print(json.dumps({"output": str(target), "samples": len(pred), "seed": args.seed,
                      "model": experiment.PRIMARY_VARIANT, "unit": "m"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
