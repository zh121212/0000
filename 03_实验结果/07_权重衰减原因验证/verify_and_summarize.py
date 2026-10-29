"""复算全部训练/验证预测，检查原文件未变，并补充相同学习率下的参数对照。"""
from pathlib import Path
import json
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import local_weight_decay_diagnosis as exp
import numpy as np
import pandas as pd
import torch


def main():
    cfg, audit = exp.context()
    receipt = exp.base.read(OUT / "completed.json")
    assert receipt["signatures"] == exp.sig_parts()
    plan = exp.base.read(OUT / "plan.json")
    assert exp.protected_files() == plan["protected_files"]
    results = list((OUT / "training_wd0").rglob("result.json"))
    assert len(results) == 14
    for path in results:
        result = exp.base.read(path)
        assert result["fingerprint"] == exp.signature()
        assert exp.base.sha256(path.parent / "best.pt") == result["checkpoint_sha256"]
        assert result["parameters"] == 102017
        epochs = pd.read_csv(path.parent / "epochs.csv")
        assert len(epochs) == result["epochs_run"] <= cfg["epochs"]
        assert epochs.loc[epochs.val_RMSE.idxmin(),"epoch"] == result["best_epoch"]
        np.testing.assert_allclose(epochs.val_RMSE.min(), result["best_val_RMSE"], rtol=0, atol=1e-12)
    metrics = pd.read_csv(OUT / "metrics.csv")
    permutations = pd.read_csv(OUT / "permutations.csv")
    attention = pd.read_csv(OUT / "attention.csv")
    gradients = pd.read_csv(OUT / "initial_gradient_comparison.csv")
    assert len(metrics) == 168 and len(permutations) == 600 and len(attention) == 40 and len(gradients) == 280
    thresholds = exp.base.read(exp.OLD / "plan.json")["thresholds_m"]
    packs = {s: exp.base.load("A", s) for s in ("train", "val")}
    prediction_count = 0
    max_difference = 0.0
    parameter_rows = []
    for decay in exp.DECAYS:
        for name in exp.MODELS:
            selection = exp.base.read(exp.selection_path(name, decay))
            assert selection["lr"] == min(selection["candidates"],key=lambda item:item["best_val_RMSE"])["lr"]
            for seed in cfg["seeds"]:
                for split, pack in packs.items():
                    file = OUT / "predictions" / f"{name}_wd{decay:g}_s{seed}" / f"{split}.csv"
                    df = pd.read_csv(file)
                    np.testing.assert_array_equal(df.target_index, pack["target_index"])
                    np.testing.assert_allclose(df.observed, pack["y"], rtol=0, atol=3e-11)
                    np.testing.assert_allclose(df.persistence, pack["persistence"], rtol=0, atol=3e-11)
                    assert np.isfinite(df.predicted).all()
                    for group, mask in exp.previous.groups(pack, thresholds).items():
                        pred = df.predicted.to_numpy()[mask]
                        score = exp.base.metrics(pred, pack["y"][mask], pack["persistence"][mask])
                        subset = metrics[(metrics.model==name)&(metrics.seed==seed)&(metrics.split==split)&(metrics.group==group)&(metrics.weight_decay==decay)]
                        assert len(subset) == 1
                        row = subset.iloc[0]
                        for key in ["MAE","MAPE","RMSE","skill_vs_persistence"]:
                            difference = abs(score[key]-row[key])
                            max_difference = max(difference,max_difference)
                            assert difference < 1e-10
                        assert row["n"] == int(mask.sum())
                    if split == "val":
                        job={"task":"A","model":name,"seed":seed,"lr":selection["lr"]}
                        result=exp.base.read(exp.run_path(job,decay)/"result.json")
                        rmse=exp.base.metrics(df.predicted.to_numpy(),pack["y"],pack["persistence"])["RMSE"]
                        np.testing.assert_allclose(rmse,result["best_val_RMSE"],rtol=0,atol=1e-11)
                    prediction_count += 1
            # 同一个seed、同一个LR、相同初始化下，比较两个衰减条件的最优权重。
            for lr in cfg["learning_rates"]:
                device=exp.base.setup(cfg,42,"cpu")
                model=exp.previous.experiment_build(name,audit["tasks"]["A"],cfg,device)
                initial={k:v.detach().clone() for k,v in model.named_parameters()}
                job={"task":"A","model":name,"seed":42,"lr":lr}
                model.load_state_dict(torch.load(exp.run_path(job,decay)/"best.pt",map_location="cpu",weights_only=True))
                parameter_rows += exp.norms(model,initial,name,42,decay,lr)
                del model
    exp.previous.save_csv(OUT/"matched_lr_parameter_norms.csv",parameter_rows)
    metric_columns=["MAE","MAPE","RMSE","skill_vs_persistence"]
    means=metrics[metrics.model!="Persistence"].groupby(["model","weight_decay","split","group"])[metric_columns].mean()
    means.reset_index().assign(MAE_mm=lambda d:d.MAE*1000,RMSE_mm=lambda d:d.RMSE*1000).to_csv(OUT/"结果均值_易读版.csv",index=False,encoding="utf-8-sig")
    metrics[metrics.model!="Persistence"].groupby(["model","weight_decay","split","group"])[metric_columns].agg(["mean","std","count"]).to_csv(OUT/"summary.csv",encoding="utf-8-sig")
    effect=[]
    val=metrics[(metrics.split=="val")&(metrics.group=="all")]
    persistence=float(val[val.model=="Persistence"].RMSE.iloc[0])
    for name in exp.MODELS:
        old=val[(val.model==name)&(val.weight_decay==.0001)].set_index("seed")
        new=val[(val.model==name)&(val.weight_decay==0)].set_index("seed")
        effect.append({"model":name,"control_RMSE_mm":float(old.RMSE.mean()*1000),"wd0_RMSE_mm":float(new.RMSE.mean()*1000),
            "RMSE_change_percent":float((new.RMSE.mean()/old.RMSE.mean()-1)*100),"paired_RMSE_wins":int((new.RMSE<old.RMSE).sum()),
            "paired_MAE_wins":int((new.MAE<old.MAE).sum()),"RMSE_wins_vs_persistence":int((new.RMSE<persistence).sum()),
            "control_MAE_mm":float(old.MAE.mean()*1000),"wd0_MAE_mm":float(new.MAE.mean()*1000)})
    aggregate_norms=[]
    parameters=pd.read_csv(OUT/"parameter_norms.csv")
    for (name,decay,seed,group),part in parameters.groupby(["model","weight_decay","seed","group"]):
        ratio=np.sqrt((part.trained_norm**2).sum()/(part.initial_norm**2).sum())
        aggregate_norms.append({"model":name,"weight_decay":decay,"seed":seed,"group":group,"trained_to_initial_norm":float(ratio)})
    exp.previous.save_csv(OUT/"parameter_group_norms.csv",aggregate_norms)
    decision={"date":"2026-10-29","status":"verified","validation_persistence_RMSE_mm":persistence*1000,"paired_effects":effect,
              "parameter_groups":json.loads(pd.DataFrame(aggregate_norms).groupby(["model","weight_decay","group"]).trained_to_initial_norm.mean().reset_index().to_json(orient="records")),
              "permutation_means":json.loads(permutations.groupby(["model","weight_decay","group"])[["delta_RMSE","prediction_shift_RMS_m"]].mean().reset_index().to_json(orient="records")),
              "initial_gradient_medians":json.loads(gradients.groupby(["model","group"])[["data_gradient_norm","decay_gradient_norm","decay_to_data_ratio"]].median().reset_index().to_json(orient="records")),
              "attention_means":json.loads(attention.groupby(["model","weight_decay"])[["mean_abs_deviation_from_uniform","normalized_entropy"]].mean().reset_index().to_json(orient="records")),
              "test_evaluated":False,"structural_changes_made":False}
    exp.dump(OUT/"decision_evidence.json",decision)
    verification={"status":"passed","original_protected_files":len(plan["protected_files"]),"original_files_unchanged":True,
                  "new_checkpoints_verified":14,"selected_checkpoint_conditions":20,"prediction_files_recomputed":prediction_count,
                  "metric_rows":168,"permutation_runs":600,"attention_rows":40,"gradient_probe_rows":280,
                  "matched_lr_seed42_cases":12,"max_metric_recalculation_difference":max_difference,"test_evaluated":False}
    exp.dump(OUT/"verification.json",verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps(effect,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
