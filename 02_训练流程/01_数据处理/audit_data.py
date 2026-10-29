"""只读检查本次选定的数据副本；不插值、不改写观测、不生成训练结果。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "01_数据集/01_源文件快照"
OUTPUT = ROOT / "03_实验结果/03_数据检查"


def main():
    manifest = json.loads((ROOT / "05_实验文档/source_manifest.json").read_text(encoding="utf-8"))
    rows, time_axes = [], []
    for record in manifest["files"]:
        if record["kind"] != "selected_dataset_snapshot":
            continue
        path = ROOT / record["snapshot"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"数据副本校验值改变：{path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        time = pd.to_datetime(frame["时间"], errors="raise")
        gates = [c for c in frame.columns if c.startswith("孔") and c.endswith("实际开度")]
        features = ["闸前水位", "闸后水位", "闸前闸后水位差", "瞬时流量"] + gates
        numbers = frame[features].apply(pd.to_numeric, errors="raise")
        n = len(frame)
        train_end, val_end = int(n * .7), int(n * .8)
        masks = {
            "train": (0, train_end),
            "validation": (train_end, val_end),
            "test": (val_end, n),
        }
        splits = {}
        for name, (start, end) in masks.items():
            splits[name] = {
                "raw_rows": end - start,
                "start": str(time.iloc[start]),
                "end": str(time.iloc[end - 1]),
                "history_12_forecast_samples": end - max(start, 12),
            }
        valid = np.isfinite(numbers.to_numpy()).all()
        continuous = bool(time.diff().iloc[1:].eq(pd.Timedelta(minutes=30)).all())
        diff_error = float((frame["闸前水位"] - frame["闸后水位"] - frame["闸前闸后水位差"]).abs().max())
        rows.append({
            "file": path.name, "source": record["source"], "sha256": digest,
            "rows": n, "columns": frame.columns.tolist(), "input_physical_columns": features,
            "gate_count": len(gates), "start": str(time.iloc[0]), "end": str(time.iloc[-1]),
            "missing_cells": int(frame.isna().sum().sum()),
            "nonfinite_input_cells": int((~np.isfinite(numbers.to_numpy())).sum()),
            "duplicate_timestamps": int(time.duplicated().sum()),
            "monotonic_time": bool(time.is_monotonic_increasing), "continuous_30min": continuous,
            "negative_gate_cells": int((frame[gates] < 0).sum().sum()),
            "min": numbers.min().to_dict(), "max": numbers.max().to_dict(),
            "difference_consistency_max_abs_error": diff_error,
            "split_draft_70_10_20": splits,
            "structure_check_passed": bool(valid and continuous and not time.duplicated().any()
                                           and not frame.isna().any().any()),
        })
        time_axes.append(time)
    cfg = json.loads((ROOT / "04_实验配置/protocol.json").read_text(encoding="utf-8"))
    if len(rows) != len(cfg["stations"]):
        raise ValueError(f"预期{len(cfg['stations'])}个站点文件，实际为{len(rows)}")
    aligned = all(time_axes[0].equals(t) for t in time_axes[1:])
    report = {
        "status": "STRUCTURE_CHECKED_ONLY",
        "note": "结构检查不等于已核实测量真实性、单位、异常值原因和水力连接关系。",
        "selected_source": manifest["selected_source"], "all_time_axes_equal": aligned,
        "files": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 选定数据的检查结果", "", f"来源：{manifest['selected_source']}", "", "仅完成结构和数值检查；未训练模型。", ""]
    for item in rows:
        lines += [f"## {item['file']}", "", f"记录数：{item['rows']}；时间：{item['start']} 至 {item['end']}。",
                  f"30分钟连续：{item['continuous_30min']}；缺失单元格：{item['missing_cells']}；重复时间：{item['duplicate_timestamps']}。",
                  f"分孔开度：{item['gate_count']}孔；物理输入：{len(item['input_physical_columns'])}列。",
                  f"水位差一致性最大误差：{item['difference_consistency_max_abs_error']:.3g}。", ""]
    lines += [f"所选站点时间轴完全相同：{aligned}。", "", "单位未经独立确认，不换算。零开度保留；极端开度只标记，不能凭大小直接删除。"]
    (OUTPUT / "数据检查.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not aligned or not all(r["structure_check_passed"] for r in rows):
        raise ValueError("结构检查未通过，详见data_audit.json")
    print(json.dumps({"status": report["status"], "files": len(rows), "time_aligned": aligned,
                      "output": str(OUTPUT / "data_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
