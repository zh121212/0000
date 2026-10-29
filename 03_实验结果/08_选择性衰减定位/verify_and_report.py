"""校验选择性衰减实验，合并三条件证据，输出最终原因报告。"""
from pathlib import Path
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import local_selective_decay_diagnosis as exp
import numpy as np
import pandas as pd
import torch


def main():
    cfg,audit=exp.context()
    receipt=exp.base.read(OUT/"completed.json")
    assert receipt["signature_parts"]==exp.signature_parts()
    assert exp.comparison.protected_files()==exp.base.read(exp.comparison.OUT/"plan.json")["protected_files"]
    selection=exp.base.read(OUT/"selection.json")
    assert selection["lr"]==min(selection["candidates"],key=lambda item:item["best_val_RMSE"])["lr"]
    exp.base.setup(cfg,42,"cpu")
    template=exp.previous.experiment_build(exp.previous.VARIANT,audit["tasks"]["A"],cfg,torch.device("cpu"))
    all_names={name for name,_ in template.named_parameters()}
    exempt_names={name for name in all_names if exp.comparison.parameter_group(name) in exp.EXEMPT}
    new_results=list((OUT/"training").rglob("result.json"))
    assert len(new_results)==7
    for file in new_results:
        record=exp.base.read(file)
        assert record["fingerprint"]==exp.signature()
        assert record["parameters"]==102017
        assert exp.base.sha256(file.parent/"best.pt")==record["checkpoint_sha256"]
        groups=exp.base.read(file.parent/"optimizer_groups.json")
        assert set(groups["exempt_names"])==exempt_names
        assert set(groups["remaining_names"])==all_names-exempt_names
        assert groups["exempt_decay"]==0 and groups["remaining_decay"]==0.0001
        assert groups["exempt_parameters"]+groups["remaining_parameters"]==102017
        state=torch.load(file.parent/"best.pt",map_location="cpu",weights_only=True)
        assert state.keys()==template.state_dict().keys()
        assert all(value.shape==template.state_dict()[key].shape for key,value in state.items())
        epochs=pd.read_csv(file.parent/"epochs.csv")
        assert len(epochs)==record["epochs_run"]<=cfg["epochs"]
        assert int(epochs.loc[epochs.val_RMSE.idxmin(),"epoch"])==record["best_epoch"]
        np.testing.assert_allclose(epochs.val_RMSE.min(),record["best_val_RMSE"],rtol=0,atol=1e-12)
    metrics=pd.read_csv(OUT/"metrics.csv")
    permutation=pd.read_csv(OUT/"permutations.csv")
    attention=pd.read_csv(OUT/"attention.csv")
    assert len(metrics)==40 and len(permutation)==150 and len(attention)==10
    thresholds=exp.base.read(exp.previous.OUT/"plan.json")["thresholds_m"]
    difference=0.0
    files=0
    for seed in cfg["seeds"]:
        for split in ["train","val"]:
            pack=exp.base.load("A",split)
            pred=pd.read_csv(OUT/"predictions"/f"s{seed}"/f"{split}.csv")
            np.testing.assert_array_equal(pred.target_index,pack["target_index"])
            np.testing.assert_allclose(pred.observed,pack["y"],rtol=0,atol=3e-11)
            np.testing.assert_allclose(pred.persistence,pack["persistence"],rtol=0,atol=3e-11)
            assert np.isfinite(pred.predicted).all()
            for group,mask in exp.previous.groups(pack,thresholds).items():
                scores=exp.base.metrics(pred.predicted.to_numpy()[mask],pack["y"][mask],pack["persistence"][mask])
                row=metrics[(metrics.seed==seed)&(metrics.split==split)&(metrics.group==group)].iloc[0]
                assert int(row["n"])==int(mask.sum())
                for key in ["MAE","MAPE","RMSE","skill_vs_persistence"]:
                    current=abs(row[key]-scores[key]);difference=max(difference,current)
                    assert current<1e-10
            if split=="val":
                job={"task":"A","model":exp.NAME,"seed":seed,"lr":selection["lr"]}
                rmse=exp.base.metrics(pred.predicted.to_numpy(),pack["y"],pack["persistence"])["RMSE"]
                np.testing.assert_allclose(rmse,exp.complete(job)["best_val_RMSE"],rtol=0,atol=1e-11)
            files+=1
    first=pd.read_csv(exp.comparison.OUT/"metrics.csv")
    control=first[first.model==exp.previous.VARIANT].copy()
    control["condition"]=np.where(control.weight_decay==0,"uniform_zero","uniform_1e-4")
    baseline=first[first.model=="Persistence"].copy();baseline["condition"]="Persistence"
    selective=metrics.copy();selective["condition"]="external_exempt"
    combined=pd.concat([control,selective,baseline],ignore_index=True)
    combined.to_csv(OUT/"三条件逐种子对照.csv",index=False,encoding="utf-8-sig")
    columns=["MAE","MAPE","RMSE","skill_vs_persistence"]
    means=combined.groupby(["condition","split","group"])[columns].mean()
    deviations=combined.groupby(["condition","split","group"])[columns].std()
    means.reset_index().assign(MAE_mm=lambda d:d.MAE*1000,RMSE_mm=lambda d:d.RMSE*1000).to_csv(OUT/"三条件结果均值.csv",index=False,encoding="utf-8-sig")
    def value(condition,group="all",metric="RMSE",split="val"):
        return float(means.loc[(condition,split,group),metric])
    def mm(condition,group="all",metric="RMSE",split="val"):
        return value(condition,group,metric,split)*1000
    def reduction(a,b,group="all",metric="RMSE"):
        return (1-value(b,group,metric)/value(a,group,metric))*100
    result=[]
    for condition in ["uniform_1e-4","uniform_zero","external_exempt"]:
        part=combined[(combined.condition==condition)&(combined.split=="val")&(combined.group=="all")].set_index("seed")
        original=combined[(combined.condition=="uniform_1e-4")&(combined.split=="val")&(combined.group=="all")].set_index("seed")
        result.append({"condition":condition,"val_RMSE_mm":float(part.RMSE.mean()*1000),"val_RMSE_std_mm":float(part.RMSE.std()*1000),
                       "val_MAE_mm":float(part.MAE.mean()*1000),"RMSE_wins_vs_original":int((part.RMSE<original.RMSE).sum()),
                       "MAE_wins_vs_original":int((part.MAE<original.MAE).sum()),"RMSE_wins_vs_persistence":int((part.RMSE<value("Persistence")).sum())})
    selected=min(result,key=lambda row:row["val_RMSE_mm"])["condition"]
    contributions=[]
    for condition in ["uniform_1e-4","uniform_zero","external_exempt"]:
        part=combined[(combined.condition==condition)&(combined.split=="val")]
        total=float((part[part.group=="all"].RMSE**2*part[part.group=="all"]["n"]).mean())
        for group in ["stable","moderate","fast"]:
            subset=part[part.group==group]
            contributions.append({"condition":condition,"group":group,"fraction_of_validation_squared_error":float((subset.RMSE**2*subset["n"]).mean()/total)})
    exp.previous.save_csv(OUT/"验证误差贡献.csv",contributions)
    verification={"status":"passed","new_checkpoints_verified":7,"optimizer_group_assignments_verified":7,
        "unchanged_backbone_and_parameter_count":True,"parameters":102017,"prediction_files_recomputed":files,
        "metric_rows":40,"permutation_runs":150,"max_metric_recalculation_difference":difference,
        "original_protected_files_unchanged":True,"test_evaluated":False}
    exp.base.write_json(OUT/"verification.json",verification)
    exp.base.write_json(OUT/"final_decision.json",{"date":"2026-10-29","conditions":result,"selection_metric":"five-seed validation RMSE",
        "selected_condition_by_metric":selected,"selected_model_family":"TimeXer_LastAnchor","selected_lr":0.001 if selected=="uniform_zero" else selection["lr"],
        "new_training_jobs_total":21,"permutation_runs_total":750,"test_evaluated":False,"architecture_changed":False,
        "reason":"coupled weight decay shrinks weak external paths; selective exemption restores paths and recovers part of accuracy without changing architecture",
        "remaining_tradeoff":"removing more regularization benefits fast changes but harms stable and/or moderate regimes",
        "module_status":"no module added: training constraint explains a verified part of the issue; calendar utility and a reliable future-state gate are not established"})
    report=f"""# 最终原因判断与后续方案

日期：2026-10-29。Verification Status：VERIFIED。
本轮新增21次训练：全网取消衰减14次、选择性衰减7次；复用前轮14次对照。共750次输入扰动，所有新训练均完成，无失败任务被省略。原公共协议、原模型结构和旧结果保持不变。
本轮只评估训练集与验证集，没有计算或查看新的测试指标。

## 原因已经明确到什么程度

**已确认的一个原因：统一的耦合权重衰减对弱梯度外生路径施加了过强的相对约束。**
优化器实际使用torch.optim.Adam的耦合衰减，梯度包含`数据梯度 + 0.0001 × 参数`。
初始交叉注意力查询/键参数上，衰减项与数据梯度范数之比的中位数，在原TimeXer约为13.5，在LastAnchor约为27.9。原设置下LastAnchor查询/键参数几乎归零；取消衰减后恢复到初始化量级，完整验证段注意力也恢复了区分。
这些证据解释了“输入已经进入嵌入，但后续预测几乎不响应”的现象。此前的日历归一化修复保证输入可见；这次定位的是训练后的路径退化，并非日历修复失效。

**选择性对照进一步支持这个定位。** 本次只让外生嵌入和交叉注意力免衰减，其他参数仍保持0.0001。模型所有参数键与形状、102,017个参数和原主干均完整保留。
该条件也恢复了外生路径，并把验证误差从{mm('uniform_1e-4'):.6f}降到{mm('external_exempt'):.6f} mm。因此不必先增加网络结构，就能修正已经确认的一部分问题。
不过，不同条件按同等预算各自选学习率、最优轮数也不同，不能把差值直接当成严格的因果贡献百分比。

## 三个训练条件的完整结果

三者都是同一个LastAnchor网络；差异仅在衰减的作用范围。学习率均用种子42搜索0.0001/0.0003/0.001，再运行其余四个种子；数据、输入、输出、损失、dropout、批量和早停规则相同。

- 全网衰减0.0001：验证RMSE **{mm('uniform_1e-4'):.6f} mm**，MAE **{mm('uniform_1e-4',metric='MAE'):.6f} mm**。
- 全网衰减0：验证RMSE **{mm('uniform_zero'):.6f} mm**，MAE **{mm('uniform_zero',metric='MAE'):.6f} mm**。RMSE较原条件降低{reduction('uniform_1e-4','uniform_zero'):.3f}%。
- 仅外生路径免衰减：验证RMSE **{mm('external_exempt'):.6f} mm**，MAE **{mm('external_exempt',metric='MAE'):.6f} mm**。RMSE较原条件降低{reduction('uniform_1e-4','external_exempt'):.3f}%。
- 持续性预测：验证RMSE **{mm('Persistence'):.6f} mm**，MAE **{mm('Persistence',metric='MAE'):.6f} mm**。

全网取消衰减和选择性衰减，五个种子的总体RMSE均优于原LastAnchor条件和持续性预测。MAE并非每个种子都改善，详见三条件逐种子对照.csv。
验证RMSE的五种子样本标准差，全网0为{deviations.loc[('uniform_zero','val','all'),'RMSE']*1000:.6f} mm，选择性衰减为{deviations.loc[('external_exempt','val','all'),'RMSE']*1000:.6f} mm。标准差只描述训练随机性，不能当成跨站点不确定性或显著性证明。

## 收益伴随的代价

- 平稳段RMSE：原条件{mm('uniform_1e-4','stable'):.6f}；全网0为{mm('uniform_zero','stable'):.6f}；选择性衰减为{mm('external_exempt','stable'):.6f} mm。
- 中等变化段RMSE：原条件{mm('uniform_1e-4','moderate'):.6f}；全网0为{mm('uniform_zero','moderate'):.6f}；选择性衰减为{mm('external_exempt','moderate'):.6f} mm。
- 快速变化段RMSE：原条件{mm('uniform_1e-4','fast'):.6f}；全网0为{mm('uniform_zero','fast'):.6f}；选择性衰减为{mm('external_exempt','fast'):.6f} mm。

选择性衰减比全网0的平稳段RMSE低{reduction('uniform_zero','external_exempt','stable'):.3f}%，但总体RMSE略高。不能写成所有状态、所有指标都提升。
当前选择规则一直是验证RMSE，不能事后因为某个条件更适合论文叙述就改选它。按这个规则，本轮总体RMSE最优的条件是 **LastAnchor + 全网weight_decay=0**，学习率0.001；选择性衰减作为保留的定位对照和更平衡的备选。

## 是否应该继续叠加模块

**本轮没有增加模块或删除任何关键结构。** 已验证收益在原结构下就能获得，首先应固定训练条件，再判断结构是否还有必要。
外生输入有响应也不等于一定有用：原TimeXer取消衰减后，打乱上游反而平均减小验证RMSE；LastAnchor中上游平均有小幅正向作用，日历收益仍不稳定。
此外，历史波动与下一步绝对变化的Spearman秩相关约为0.22，验证段中间波动分组也不是单调关系。它不足以支持直接加入“历史平稳就强制少修正”的门控。
因此，不能把现在的剩余问题直接归因于缺少新模块，也不应通过无依据叠加来追求当前验证段的分数。
窗口归一化去掉恒定开度的绝对水平是另一个已观察到的表示限制；本轮没有同时改它，也还没有证明补回这些绝对值会带来预测收益。

## 当前可执行方案

1. 保留LastAnchor输出基准：最后实测水位加上模型预测变化量，所有原主干结构保留。
2. 当前以验证RMSE选型时，使用本轮已训练的全网weight_decay=0、学习率0.001、五种子版本；平稳段代价必须同时披露。
3. 选择性衰减版本单独保留，供更重视平稳段与训练稳定性的应用比较，不能偷偷替换主指标选择规则。
4. 不改原七模型公共配置、不覆盖原Baseline、不改原七模型测试排名。未来进行新结构比较时，原TimeXer与改进模型必须使用相同训练条件和调参预算，才能区分结构收益和训练规则收益。
5. 现在固定方案，在后续站点或新时间段做确认。前轮3.693 mm测试RMSE属于旧的全网0.0001版本，**不能当作本轮weight_decay=0版本的测试成绩**。

## 文件与复现

- 根目录local_weight_decay_diagnosis.py：14次全网衰减对照及梯度、注意力、输入扰动。
- 根目录local_selective_decay_diagnosis.py：7次选择性衰减定位；optimizer_groups.json记录每个参数的实际分组。
- 07_权重衰减原因验证/verify_and_summarize.py：第一阶段指标与权重校验。
- 本目录verify_and_report.py：第二阶段校验及最终报告重建。
- 本目录三条件结果均值.csv、三条件逐种子对照.csv：全部条件、种子、分组结果，水位误差同时给出m和mm。
- 07目录training_wd0/01_对比实验/runs/A_TimeXer_LastAnchor_lr0.001_s{{seed}}/best.pt：当前验证RMSE最优条件的五个权重。
- 本目录training/01_对比实验/runs/A_TimeXer_LastAnchor_PathDecay_lr0.0003_s{{seed}}/best.pt：选择性衰减五个正式权重。
- 两目录均保留训练日志、逐轮指标、训练/验证预测、所有学习率候选权重及校验凭据。

```powershell
& 'D:/anaconda3/envs/rl_case1/python.exe' -X utf8 './03_实验结果/07_权重衰减原因验证/verify_and_summarize.py'
& 'D:/anaconda3/envs/rl_case1/python.exe' -X utf8 './03_实验结果/08_选择性衰减定位/verify_and_report.py'
```

## 统计解释检查（11/11）

总体与分组方向均披露；五种子不冒充五站点；单一时间段存在适用边界；未按测试结果筛样本或改控制变量；保留持续性对照；不选最差到最好种子制造改善；21次训练完整无失败省略；不作多重筛选后的显著性宣称；明确探索性分析与阶段顺序；区分优化器受控干预和水文因果；未来真实变化仅用于评分、不进入模型或在线门控。
"""
    (OUT/"最终原因与改进方案.md").write_text(report,encoding="utf-8")
    exp.base.write_json(OUT/"progress.json",{"phase":"completed_and_verified","completed_jobs":7,"total_jobs":7,"test_evaluated":False})
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("SELECTED BY VALIDATION RMSE",selected)


if __name__=="__main__":
    main()
