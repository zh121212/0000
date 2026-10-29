"""复算本地三个实验的结果，检查原实验未变，并生成中文交付报告。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(pred, true, persistence):
    return {"MAE": float(np.abs(pred-true).mean()),
            "MAPE": float(np.abs((pred-true)/true).mean()*100),
            "RMSE": float(np.sqrt(np.mean((pred-true)**2))),
            "skill_vs_persistence": float(1-np.mean((pred-true)**2)/np.mean((persistence-true)**2)),
            "bias_m": float((pred-true).mean())}


def main():
    complete = read(OUT / "completed.json")
    original = read(ROOT / "03_实验结果/运行记录/fit_complete.json")
    cfg = read(ROOT / "04_实验配置/protocol.json")
    sources = [ROOT / "run.py", ROOT / "04_实验配置/protocol.json"]
    sources += sorted((ROOT / "02_训练流程").rglob("*.py"))
    sources += sorted((ROOT / "01_数据集/01_源文件快照").glob("*.csv"))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    baseline_signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert baseline_signature == original["fingerprint"] == complete["signatures"]["baseline"]
    assert sha(ROOT / "local_three_experiments.py") == complete["signatures"]["experiment_script_sha256"]
    audit = read(ROOT / "01_数据集/03_划分打包/packing_audit.json")
    for relative, digest in audit["output_sha256"].items():
        assert sha(ROOT / "01_数据集/03_划分打包" / relative) == digest
    original_runs = list((ROOT / "03_实验结果/01_对比实验/runs").glob("*/result.json"))
    for path in original_runs:
        result = read(path)
        assert result["fingerprint"] == baseline_signature
        assert sha(path.parent / "best.pt") == result["checkpoint_sha256"]
    local_runs = list((OUT / "experiment3/01_对比实验/runs").glob("*/result.json"))
    assert len(local_runs) == 14 and len(original_runs) == 49
    experiment_signature = hashlib.sha256(json.dumps(complete["signatures"], sort_keys=True).encode()).hexdigest()
    for path in local_runs:
        result = read(path)
        assert result["fingerprint"] == experiment_signature
        assert sha(path.parent / "best.pt") == result["checkpoint_sha256"]
        assert result["parameters"] == 102017
        epochs = pd.read_csv(path.parent / "epochs.csv")
        assert len(epochs) == result["epochs_run"] <= cfg["epochs"]
        assert int(epochs.loc[epochs.val_RMSE.idxmin(), "epoch"]) == result["best_epoch"]
        np.testing.assert_allclose(epochs.val_RMSE.min(), result["best_val_RMSE"], rtol=0, atol=1e-12)
    frame = pd.read_csv(OUT / "experiment3_metrics.csv")
    assert len(frame) == 132
    thresholds = read(OUT / "plan.json")["thresholds_m"]
    verified_predictions = 0
    for name in ("TimeXer", "TimeXer_LastAnchor"):
        selection = read(OUT / f"selection_{name}.json")
        assert min(selection["candidates"], key=lambda item: item["best_val_RMSE"])["lr"] == selection["lr"]
        for seed in cfg["seeds"]:
            dest = OUT / "experiment3/01_对比实验/runs" / f"A_{name}_lr{selection['lr']:g}_s{seed}"
            for split in ("train", "val", "test"):
                prediction = pd.read_csv(dest / f"{split}_predictions.csv")
                with np.load(ROOT / f"01_数据集/03_划分打包/A/{split}.npz") as data:
                    true, persistence = data["y"], data["persistence"]
                    np.testing.assert_array_equal(prediction.target_index, data["target_index"])
                    np.testing.assert_allclose(prediction.observed, true, rtol=0, atol=3e-11)
                    np.testing.assert_allclose(prediction.persistence, persistence, rtol=0, atol=3e-11)
                    assert np.isfinite(prediction.predicted).all()
                    delta = np.abs(true-persistence)
                    masks = {"all": np.ones(len(true), dtype=bool), "stable": delta <= thresholds[0],
                             "moderate": (delta > thresholds[0]) & (delta <= thresholds[1]), "fast": delta > thresholds[1]}
                    for group, mask in masks.items():
                        row = frame[(frame.model == name) & (frame.seed == seed) & (frame.split == split) & (frame.group == group)].iloc[0]
                        assert row["n"] == int(mask.sum())
                        recomputed = metrics(prediction.predicted.to_numpy()[mask], true[mask], persistence[mask])
                        for key, value in recomputed.items():
                            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-10)
                    if split == "val":
                        np.testing.assert_allclose(metrics(prediction.predicted.to_numpy(), true, persistence)["RMSE"], read(dest / "result.json")["best_val_RMSE"], rtol=0, atol=1e-11)
                verified_predictions += 1
    first = pd.read_csv(OUT / "experiment1_eval_metrics.csv")
    assert len(first) == 288
    permutation = pd.read_csv(OUT / "experiment2_permutation.csv")
    variant_permutation = pd.read_csv(OUT / "experiment3_permutation.csv")
    assert len(permutation) == len(variant_permutation) == 150
    integrity = {"status": "passed", "baseline_fingerprint_unchanged": True, "original_checkpoints_verified": 49,
                 "local_checkpoints_verified": 14, "prediction_files_recomputed": verified_predictions,
                 "metric_rows": len(frame), "original_eval_rows": len(first), "permutation_runs_original": len(permutation),
                 "permutation_runs_variant": len(variant_permutation), "data_packs_verified": len(audit["output_sha256"]),
                 "same_parameter_count": 102017, "original_protocol_unchanged": True, "test_used_for_selection": False}
    write(OUT / "verification.json", integrity)
    write(OUT / "progress.json", {"phase": "completed_and_verified", "completed_jobs": 14, "total_jobs": 14,
                                  "failed": [], "verification": integrity})

    # 单行表头，水位误差同时给出 m 与 mm，便于直接阅读。
    means = frame.groupby(["model", "split", "group"], sort=False)[["MAE", "MAPE", "RMSE", "skill_vs_persistence"]].mean()
    deviations = frame.groupby(["model", "split", "group"], sort=False)[["MAE", "RMSE"]].std()
    readable = means.reset_index()
    for metric in ("MAE", "RMSE"):
        readable[metric + "_mm"] = readable[metric] * 1000
    readable.to_csv(OUT / "结果均值_易读版.csv", index=False, encoding="utf-8-sig")
    existing = pd.read_csv(ROOT / "03_实验结果/01_对比实验/all_seed_metrics.csv")
    existing["result_source"] = "original_cloud"
    added = frame[(frame.model == "TimeXer_LastAnchor") & (frame.split == "test") & (frame.group == "all")].copy()
    added["result_source"] = "local_LastAnchor"
    comparison = pd.concat([existing, added], ignore_index=True)
    comparison.groupby("model")[["MAE", "MAPE", "RMSE", "skill_vs_persistence"]].agg(["mean", "std", "count"]).to_csv(OUT / "全部模型测试对照.csv", encoding="utf-8-sig")
    def v(name, split="test", group="all", metric="RMSE"):
        return float(means.loc[(name, split, group), metric])
    def pct(old, new):
        return 100 * (old-new) / old
    old, new, persistence = "TimeXer", "TimeXer_LastAnchor", "Persistence"
    best_existing = existing.groupby("model")["RMSE"].mean().idxmin()
    best_rmse = existing.groupby("model")["RMSE"].mean().min()
    val_wins = int((frame[(frame.model == new) & (frame.split == "val") & (frame.group == "all")].RMSE < v(persistence, "val")).sum())
    test_wins = int((frame[(frame.model == new) & (frame.split == "test") & (frame.group == "all")].RMSE < v(persistence)).sum())
    train_eval = first[(first.model == old) & (first.split == "train") & (first.group == "all")].RMSE.mean()
    perm_mean = permutation.groupby("group")[["delta_RMSE", "prediction_shift_RMS_m"]].mean()
    report = f"""# 本地三个实验：原因判断与最终改进方案

实验状态：完成并通过完整性验证。日期：2026-10-27。
验证范围：本地复现35个已有最优权重；原版/改进版各7次训练，共14次；原版150次分组扰动，并对改进版重复150次扰动。
研究对象：真实数据、12步历史输入、30分钟后水位的单步预测。水位误差正文使用mm，原始结果使用m。改进模型不是五种子集成，均值是五次独立训练的指标平均。

## 完成结论

当前保留的最小改进方案为 **TimeXer_LastAnchor（以最后实测水位作为输出基准）**。
它修正了当前任务下输出基准与强持续性先验之间的不匹配，实验支持这是一项可修正的误差来源；不是证明了全部误差都来自这一因素。
原模型主干、输入字段和训练规则均保留，没有新增可训练模块，参数数量仍为102,017。
该方案在训练、验证和当前测试段均优于持续性预测，但上游/日历利用不足与平稳段的剩余误差尚未解决。

## 实验1：是否主要是过拟合

使用原7个模型的35个最优验证权重，在eval模式关闭训练用随机失活，统一计算原始水位误差；不把训练目标中的其他项当成水位误差。
原TimeXer训练RMSE为{train_eval*1000:.6f} mm，持续性预测为{v(persistence,'train')*1000:.6f} mm。
原TimeXer验证RMSE为{v(old,'val')*1000:.6f} mm，持续性预测为{v(persistence,'val')*1000:.6f} mm。

判断：存在部分训练后期验证误差回升，但所用权重来自最优验证轮，不能将最终问题直接归为严重过拟合。
更关键的是原TimeXer连训练集的总体RMSE也没有超过持续性预测，因此“只要控制过拟合就能解决”不受这些结果支持。
验证段的持续性误差本身就更高，说明不同时间段难度不同；训练/验证绝对误差差距不能单独作为过拟合证据。

## 实验2：哪些输入实际影响预测

冻结原TimeXer五个种子的权重，只在验证集上按整段历史窗口置换：上游8个物理变量一起、本站6个物理变量一起、日历3个变量一起。每组10次，所有种子使用相同置换顺序。
同组内变量关系和12步内部时序保留，组间关联会被破坏，因此这是依赖诊断，不能直接解释为真实水文因果。

原TimeXer平均预测变化的均方根：

- 上游物理信息：{perm_mean.loc['upstream_physical','prediction_shift_RMS_m']*1000:.9f} mm。
- 日历信息：{perm_mean.loc['calendar','prediction_shift_RMS_m']*1000:.9f} mm。
- 本站历史：{perm_mean.loc['local_physical','prediction_shift_RMS_m']*1000:.6f} mm；验证RMSE平均增加{perm_mean.loc['local_physical','delta_RMSE']*1000:.6f} mm。

判断：当前模型实际主要依赖本站历史。本站扰动同时破坏输出水位基准，所以这不能证明本站每个变量都有效。
进一步只读检查发现：输入扰动能改变外生变量嵌入，但编码层和输出几乎不响应；五种子各自前64个验证窗口的交叉注意力均接近1/16的均匀分配。
第一层交叉注意力查询、键、值投影权重的范数，五种子平均约为初始化的1%左右；这是路径退化的证据，具体见parameter_norms和attention文件。

Feature Suppression（特征抑制，即输入虽然存在，但在模型内部影响被削弱）在这里存在“功能上近乎忽略外生信息”的现象。
但还没有证明这些输入对30分钟预测必然有额外价值，也没有通过正则化对照实验证明权重衰减就是原因，因此不能断言已经证明“有用特征受到抑制”。
此前日历修复保证了日历值可见；本次发现的是训练后的利用不足，两者不同。
对改进版重复相同150次扰动后，上游/日历仍几乎无影响，所以本次提升不是信息融合能力恢复带来的。

## 实验3：验证最后实测水位基准

原版最终目标输出：预测头输出 × 窗口标准差 + 窗口水位均值。
改进版最终目标输出：预测头输出 × 窗口标准差 + 最后实测水位。
实现等价式：`new_prediction = original_prediction - historical_target_mean + last_target`。
该等价式在同一训练标准化坐标内计算，之后按原流程恢复m。预测头需要重新训练，不能把新公式直接套到原权重当作正式改进结果。

保留注意力、分块、全局标记、日历输入、两层编码器、原参数结构及全部17个输入变量。
变体只增加一次固定的输出修正，不增加可训练参数；未做严格运算量/推理速度基准，不能由训练耗时推断结构加速。

每个模型都按seed42搜索0.0001、0.0003、0.001，按验证RMSE选学习率，再训练其余4个种子。
原版选0.0001，改进版选0.001；搜索范围与预算相同，各自选择不同学习率不构成不公平修改。
其余设置完全相同：原数据包、训练段标准化、70/10/20时间划分、12步历史、单步输出、Adam、batch64、最多100轮、patience10、dropout0.1、weight_decay0.0001、梯度裁剪5。
两分支在同一本地GPU上同时执行，均使用原训练函数；完成所有训练与选参后才计算本次测试指标。

### 总体结果

- 测试RMSE：原版{v(old)*1000:.6f} mm；改进版{v(new)*1000:.6f} mm；持续性预测{v(persistence)*1000:.6f} mm。
- 改进版测试RMSE五种子样本标准差：{deviations.loc[(new,'test','all'),'RMSE']*1000:.6f} mm。
- 测试MAE：原版{v(old,metric='MAE')*1000:.6f} mm；改进版{v(new,metric='MAE')*1000:.6f} mm；持续性预测{v(persistence,metric='MAE')*1000:.6f} mm。
- 测试MAPE：原版{v(old,metric='MAPE'):.9f}%；改进版{v(new,metric='MAPE'):.9f}%。
- 相对原版：RMSE降低{pct(v(old),v(new)):.3f}%，MAE降低{pct(v(old,metric='MAE'),v(new,metric='MAE')):.3f}%。
- 相对持续性预测：RMSE降低{pct(v(persistence),v(new)):.3f}%，MAE降低{pct(v(persistence,metric='MAE'),v(new,metric='MAE')):.3f}%。
- 改进版平均技能分数{v(new,metric='skill_vs_persistence'):.6f}，即平均MSE比持续性预测降低{v(new,metric='skill_vs_persistence')*100:.3f}%，不是准确率。
- 验证RMSE：原版{v(old,'val')*1000:.6f} mm；改进版{v(new,'val')*1000:.6f} mm；持续性预测{v(persistence,'val')*1000:.6f} mm。改进版相对持续性仅改善{pct(v(persistence,'val'),v(new,'val')):.3f}%，收益较小。
- 五种子中，改进版验证集{val_wins}/5、测试集{test_wins}/5的总体RMSE优于持续性预测，且均优于相同种子的原版。
- 原有7个学习模型中测试RMSE最低的是{best_existing}：{best_rmse*1000:.6f} mm。改进版比它低{pct(best_rmse,v(new)):.3f}%；这属于当前测试段的描述性比较。

### 分时段收益与代价

分组阈值仅由训练标签的一步绝对变化的50%与90%分位数确定：{thresholds[0]*1000:.6f} mm与{thresholds[1]*1000:.6f} mm。
这是评估分组，不是水文标准，也不是传感器精度；需要真实未来变化才能分组，不能用它在线决定采用哪个模型。

- 平稳段测试RMSE：原版{v(old,group='stable')*1000:.6f} → 改进版{v(new,group='stable')*1000:.6f} mm，下降{pct(v(old,group='stable'),v(new,group='stable')):.3f}%；但仍差于持续性预测的{v(persistence,group='stable')*1000:.6f} mm。
- 中等变化段测试RMSE：{v(old,group='moderate')*1000:.6f} → {v(new,group='moderate')*1000:.6f} mm；MAE却从{v(old,group='moderate',metric='MAE')*1000:.6f}升到{v(new,group='moderate',metric='MAE')*1000:.6f} mm，增加{100*(v(new,group='moderate',metric='MAE')/v(old,group='moderate',metric='MAE')-1):.3f}%。不能写成所有场景、所有指标都提升。
- 快速变化段测试RMSE：{v(old,group='fast')*1000:.6f} → {v(new,group='fast')*1000:.6f} mm，也优于持续性预测的{v(persistence,group='fast')*1000:.6f} mm，没有出现预先担心的快速变化误差上升。

## 最终方案及适用边界

当前采用独立变体TimeXer_LastAnchor作为后续改进的起点：保持主干，通过最后实测水位加上学习到的变化量完成预测。
结果支持优先修正预测基准，暂不支持通过随机增加模块或单纯扩大模型来解决问题。
这不是已经解决上游/日历利用不足的方案；也没有充分证据将退化唯一归因于过拟合或正则化。
未改变原工程默认入口，没有替换或覆盖任何Baseline。原版与改进版的完整训练记录、权重和预测均分别保留。

当前测试集曾在本次实验之前被用于问题诊断，因此本次测试提升属于探索结果；验证选参过程未使用本次测试结果，也不能消除之前已经看过测试集的影响。
方案与学习率现在应固定，再用后续站点或新时间段检验；五种子只反映训练随机性，不等同五个独立站点，不能据此宣称跨站点有效或统计显著。

## 文件与使用

- 独立入口：项目根目录`local_three_experiments.py`，LastAnchor类为唯一模型行为变动。
- 复现三个实验：在项目根目录执行 `& 'D:/anaconda3/envs/rl_case1/python.exe' -X utf8 -u './local_three_experiments.py' --stage all`。
- 同版本完整训练会校验后复用，不覆盖已有权重；未完成的训练会报错而非覆盖。
- 复算结果：在项目根目录执行 `& 'D:/anaconda3/python.exe' -X utf8 './03_实验结果/06_本地原因验证/verify_results.py'`。
- 易读指标：`结果均值_易读版.csv`；原始逐种子/分组指标：`experiment3_metrics.csv`。
- 权重：`experiment3/01_对比实验/runs/A_TimeXer_LastAnchor_lr0.001_s{{seed}}/best.pt`。
- 每个正式种子目录均有train/val/test_predictions.csv、epochs.csv、result.json、console.log。
- 证据文件：experiment1_eval_metrics.csv、experiment2_permutation.csv、experiment2_information_path.json、experiment2_attention_all_seeds.json、experiment2_parameter_norms.csv、experiment3_permutation.csv。
- 校验凭据：verification.json；原49个权重、本地14个权重、4个数据包及30份预测文件均通过检查。

## 统计解释检查（11/11）

1. 分组与总体方向：披露中等段MAE回退及平稳段仍落后持续性预测，避免总体均值掩盖差异。
2. 群体到个体推断：五种子不代表多个站点，未作跨站点结论。
3. 选择偏差：单站点时间段，外部适用性有限。
4. 条件选择偏差：未按测试成绩剔除种子、样本或控制变量；状态分组仅用于评分。
5. 基准比例：同时给出持续性预测和原模型对照；不把低MAPE解释成高增益。
6. 回归均值：未从最差种子选起点做前后对比，报告全部五种子。
7. 幸存者偏差：14次训练完整，无失败任务被省略。
8. 多重比较：披露全部预定实验与分组，无显著性筛选。
9. 分析自由度：方案源于既有测试诊断，明确标为探索，未宣称预注册确认性实验。
10. 因果界限：输出基准受控对照支持该改动有效；特征扰动不证明水文因果或正则化原因。
11. 时间方向：模型只读取历史，评估分组使用未来标签仅用于评分，不用于模型输入或决策。
"""
    (OUT / "实验结论与最终方案.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, ensure_ascii=False, indent=2))
    print("REPORT", OUT / "实验结论与最终方案.md")


if __name__ == "__main__":
    main()
