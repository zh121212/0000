"""将各模型五个种子的真实测试指标汇总为同一张表。"""
from pathlib import Path
import numpy as np
import pandas as pd


def save_unified_results(comparison, cfg, output):
    unified = []
    for task in cfg["tasks"]:
        for name in cfg["models"]:
            group = [row for row in comparison if row["task"] == task and row["model"] == name]
            if len(group) != len(cfg["seeds"]) or {row["seed"] for row in group} != set(cfg["seeds"]):
                raise ValueError(f"种子结果不齐或重复：{task}/{name}")
            for row in sorted(group, key=lambda item: cfg["seeds"].index(item["seed"])):
                unified.append({"任务": task, "模型": name, "记录类型": "单次", "种子": row["seed"],
                                "学习率": row["lr"], "测试MAE": row["MAE"], "测试MAPE(%)": row["MAPE"], "测试RMSE": row["RMSE"],
                                "技能分数": row["skill_vs_persistence"], "训练秒数": row["training_seconds"]})
            for label, ddof in [("均值", None), ("标准差", 1)]:
                entry = {"任务": task, "模型": name, "记录类型": label, "种子": None, "学习率": group[0]["lr"]}
                for source, dest in [("MAE","测试MAE"),("MAPE","测试MAPE(%)"),("RMSE","测试RMSE"),("skill_vs_persistence","技能分数"),("training_seconds","训练秒数")]:
                    values = [row[source] for row in group]
                    entry[dest] = (float(np.mean(values) if ddof is None else np.std(values, ddof=ddof))
                                   if all(v is not None for v in values) else None)
                unified.append(entry)
        baseline = next(row for row in comparison if row["task"] == task and row["model"] == "Persistence")
        unified.append({"任务": task, "模型": "Persistence", "记录类型": "参考", "种子": None,
                        "学习率": None, "测试MAE": baseline["MAE"], "测试MAPE(%)": baseline["MAPE"], "测试RMSE": baseline["RMSE"],
                        "技能分数": baseline["skill_vs_persistence"], "训练秒数": 0.})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unified).to_csv(output, index=False, encoding="utf-8-sig")
    return unified
