"""构建真实历史窗口，先按时间划分，只在训练段估计标准化参数。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKS = ROOT / "01_数据集/03_划分打包"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configuration():
    return json.loads((ROOT / "04_实验配置/protocol.json").read_text(encoding="utf-8"))


def prepare():
    cfg = configuration()
    frames, checksums = {}, {}
    for station, filename in cfg["stations"].items():
        path = ROOT / cfg["source_subdir"] / filename
        checksums[filename] = sha256(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        dates = pd.to_datetime(df["时间"], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"时间无效：{filename}")
        if not dates.diff().iloc[1:].eq(pd.Timedelta(minutes=cfg["frequency_minutes"])).all():
            raise ValueError(f"采样间隔不连续：{filename}")
        frames[station] = df.set_index(pd.DatetimeIndex(dates))
    axes = [df.index for df in frames.values()]
    if not all(axes[0].equals(axis) for axis in axes[1:]):
        raise ValueError("所选站点时间不对齐；禁止静默丢弃时间")
    grid, length = axes[0], cfg["seq_len"]
    n = len(grid)
    train_end, val_end = int(n * cfg["train_ratio"]), int(n * cfg["validation_end_ratio"])
    audit = {"protocol": cfg["protocol"], "raw_sha256": checksums,
             "protocol_sha256": sha256(ROOT / "04_实验配置/protocol.json"),
             "prepare_sha256": sha256(__file__), "rows": n, "train_end": train_end,
             "validation_end": val_end, "tasks": {}}
    previous_path = PACKS / "packing_audit.json"
    if previous_path.exists():
        old = json.loads(previous_path.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "prepare_sha256", "raw_sha256"):
            if old[key] != audit[key]:
                raise ValueError("数据或协议已变化，不能覆盖已打包实验；请另建版本")
        for relative, digest in old["output_sha256"].items():
            if sha256(PACKS / relative) != digest:
                raise ValueError(f"数据包改变：{relative}")
        return old
    calendar = np.column_stack([(grid.hour * 2 + grid.minute // 30) / 47.,
                                grid.dayofweek / 6., ((grid.day - 1) // 7) / 4.])
    for task, (up_name, local_name) in cfg["tasks"].items():
        up, local = frames[up_name], frames[local_name]
        gate_columns = [c for c in up if c.startswith("孔") and c.endswith("实际开度")]
        local_gates = [c for c in local if c.startswith("孔") and c.endswith("实际开度")]
        upper_columns = cfg["physical_fields"] + gate_columns
        # 为TimeXer保留目标水位为最后通道；仅调整顺序，不删除本站其他历史字段。
        local_columns = [c for c in cfg["physical_fields"] + local_gates if c != "闸前水位"] + ["闸前水位"]
        physical = np.column_stack([up[upper_columns], local[local_columns]])
        if not np.isfinite(physical).all() or (up[gate_columns].to_numpy() < 0).any() or (local[local_gates].to_numpy() < 0).any():
            raise ValueError(f"{task}存在无效观测；需明确处理后再打包")
        mean, std = physical[:train_end].mean(0), physical[:train_end].std(0)
        std = np.where(std > 0, std, 1.)
        scaled = (physical - mean) / std
        up_physical_count = len(upper_columns)
        up_channels = up_physical_count + 3
        features = np.column_stack([scaled[:, :up_physical_count], calendar, scaled[:, up_physical_count:]])
        physical_names = [f"{up_name}:{c}" for c in upper_columns] + [f"{local_name}:{c}" for c in local_columns]
        feature_names = [f"{up_name}:{c}" for c in upper_columns] + cfg["calendar_features"] + [f"{local_name}:{c}" for c in local_columns]
        target = np.arange(length, n, dtype=np.int64)
        history = target[:, None] + np.arange(-length, 0)
        # 逐样本检查：所有实测输入均严格早于标签时刻。
        assert (history[:, -1] == target - 1).all() and (history < target[:, None]).all()
        x = features[history].astype(np.float32)
        y = physical[target, -1].astype(np.float64)
        persistence = physical[target - 1, -1]
        dest = PACKS / task
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "scaler.npz", physical_mean=mean, physical_std=std,
                            y_mean=mean[-1], y_std=std[-1])
        masks = {"train": target < train_end, "val": (target >= train_end) & (target < val_end), "test": target >= val_end}
        splits = {}
        for split, mask in masks.items():
            np.savez_compressed(dest / f"{split}.npz", x=x[mask], y=y[mask],
                                persistence=persistence[mask], target_index=target[mask],
                                input_end_index=history[mask, -1], time=grid[target[mask]].to_numpy(dtype="datetime64[ns]"))
            splits[split] = {"samples": int(mask.sum()), "start": str(grid[target[mask][0]]), "end": str(grid[target[mask][-1]])}
        # 三处独立检查字段和标签对应关系，不根据模型成绩修改数据。
        for row in (0, train_end - length, len(target) - 1):
            ti = target[row]
            np.testing.assert_allclose(x[row, :, -1].astype(float) * std[-1] + mean[-1], local["闸前水位"].iloc[ti-length:ti], rtol=0, atol=1e-6)
            assert y[row] == local["闸前水位"].iloc[ti]
            assert persistence[row] == local["闸前水位"].iloc[ti-1]
        audit["tasks"][task] = {"upstream": up_name, "target_station": local_name,
                                "features": feature_names, "up_channels": up_channels,
                                "local_channels": len(local_columns), "upstream_flow_index": upper_columns.index("瞬时流量"),
                                "physical_scaler_features": physical_names,
                                "total_channels": len(feature_names), "target_channel": len(feature_names)-1,
                                "time_feature_indices": list(range(up_channels-3, up_channels)),
                                "splits": splits, "scaler_fit_indices": [0, train_end-1],
                                "input_offsets": [-length, -1], "label_offset": 0}
    audit["output_sha256"] = {p.relative_to(PACKS).as_posix(): sha256(p) for p in sorted(PACKS.rglob("*.npz"))}
    write_json(previous_path, audit)
    return audit


def load(task, split, limit=None):
    with np.load(PACKS / task / f"{split}.npz", allow_pickle=False) as pack:
        values = {name: pack[name] for name in pack.files}
    return {name: value[:limit] for name, value in values.items()} if limit else values


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
