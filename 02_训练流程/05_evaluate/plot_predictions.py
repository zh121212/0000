"""以全部真实测试预测画实测/预测水位曲线；固定种子42作横向总览。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_prediction_figures(cfg, task, prediction_files, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    reference = None
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            path = Path(prediction_files[name][seed])
            frame = pd.read_csv(path, parse_dates=["time"])
            if not np.isfinite(frame[["observed", "predicted"]].to_numpy()).all():
                raise ValueError(f"预测文件含无效值：{path}")
            if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
                raise ValueError(f"预测时间顺序错误：{path}")
            if not frame["time"].diff().iloc[1:].eq(pd.Timedelta(minutes=cfg["frequency_minutes"])).all():
                raise ValueError(f"预测时间不连续：{path}")
            if reference is None:
                reference = frame
            else:
                if not frame["time"].equals(reference["time"]):
                    raise ValueError("各模型预测时间不同，不能横向比较")
                np.testing.assert_array_equal(frame["observed"], reference["observed"])
            frames[name, seed] = frame
    elapsed_days = (reference["time"] - reference["time"].iloc[0]).dt.total_seconds().to_numpy() / 86400
    lower = min(float(frame[["observed", "predicted"]].min().min()) for frame in frames.values())
    upper = max(float(frame[["observed", "predicted"]].max().max()) for frame in frames.values())
    padding = max((upper - lower) * .06, .001)
    ylim = (lower - padding, upper + padding)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
             "axes.spines.right": False, "axes.linewidth": .8, "pdf.fonttype": 42,
             "svg.fonttype": "none", "legend.frameon": False, "path.simplify": False}

    def panel(ax, frame):
        ax.plot(elapsed_days, frame["observed"], color="#222222", linewidth=.9, label="Observed", zorder=2)
        ax.plot(elapsed_days, frame["predicted"], color="#d62728", linewidth=.8,
                linestyle="--", label="Predicted", zorder=3)
        ax.set_xlim(elapsed_days[0], elapsed_days[-1])
        ax.set_ylim(*ylim)
        ax.set_ylabel("Water level (m)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=5)
        ax.legend(loc="best", fontsize=7, handlelength=2)

    generated = []

    def export(fig, stem):
        for extension in ("png", "pdf", "svg"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
            generated.append(path.name)
        plt.close(fig)

    with plt.rc_context(style):
        for (name, seed), frame in frames.items():
            fig, ax = plt.subplots(figsize=(6.2, 3.5), layout="constrained")
            panel(ax, frame)
            ax.set_xlabel("Time since test start (days)")
            ax.set_title(f"{name} | seed {seed} | 30-min ahead", fontsize=10)
            export(fig, f"{task}_{name}_seed{seed}")
        # 先固定参考种子，再看测试曲线；不从五个种子中挑最好的一次。
        reference_seed = 42
        fig, axes = plt.subplots(1, len(cfg["models"]), figsize=(3.15 * len(cfg["models"]), 3.25),
                                 squeeze=False, layout="constrained")
        for index, name in enumerate(cfg["models"]):
            ax = axes[0, index]
            panel(ax, frames[name, reference_seed])
            ax.set_xlabel(f"Time (days)\n({chr(97 + index)}) {name}")
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Observed vs predicted water level | Task {task} | seed 42 | 30-min ahead", fontsize=11)
        export(fig, f"{task}_all_models_seed42")
    metadata = {"task": task, "reference_seed": 42, "all_seeds": cfg["seeds"], "n_test_samples": len(reference),
                "time_start": str(reference["time"].iloc[0]), "time_end": str(reference["time"].iloc[-1]),
                "water_level_unit": "m", "shared_y_limits": list(ylim),
                "processing": "All test points; no smoothing, resampling or selective window; single-seed curves, not an ensemble.",
                "source_files": {name: {str(seed): str(path) for seed, path in files.items()}
                                 for name, files in prediction_files.items()}, "generated_files": generated}
    (output / "figure_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
