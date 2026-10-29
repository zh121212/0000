"""TimeXer主线改进实验入口（任务A）。

不修改原工程任何文件：原 train.py 以模块方式导入，只在本进程内替换模型构造函数，
并把需要"外部通路免衰减"的变体的 Adam 参数改成两组（优化器类型、学习率、衰减系数不变）。
七个基线不重训，直接读取v1结果，并用v1权重重新推理验证集，核对与v1记录一致后再复用。

阶段（按顺序执行，每一步都可重复运行，已完成的训练会校验后复用，不覆盖）：
  plan      写定变体清单与判据（plan.json），之后不允许修改
  selftest  随机输入Forward Test + 用新代码加载06/08/11轮旧权重的复现检查
  fit       训练V1–V7：种子42试3个学习率，按验证RMSE选定，再补4个种子；共49次
  evaluate  只在训练集、验证集上计算指标，含基线TimeXer(V0)
  judge     按plan.json判据判定C1/C2/C3，给出主方案；附D7、D8诊断
  freeze    写frozen.json（代码、权重哈希、主方案），之后才允许看测试集
  test      开发测试集评估一次；bootstrap与DM检验
用法：python -X utf8 07_TimeXer改进主线/run_improvement.py --stage plan
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location("baseline_training", ROOT / "02_训练流程/04_训练模型/train.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # 原训练代码，只导入不修改
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import timexer_variants as tv  # noqa: E402

OUT = ROOT / "03_实验结果/12_TimeXer改进主线"
V1_RESULTS = ROOT / "03_实验结果"
V1_RECEIPT = base.read(V1_RESULTS / "运行记录/fit_complete.json")
TASK = "A"
LABELS = {"TimeXer": "V0", "TimeXer_PD": "V1", "TimeXer_LA": "V2", "TimeXer_LA_PD": "V3",
          "TimeXer_LA_PD_C3Raw": "V4", "TimeXer_LA_PD_C3Zero": "V5", "TimeXer_LA_PD_C3Norm": "V6",
          "TimeXer_LA_C3Raw": "V7"}
VARIANT_NAMES = list(tv.VARIANTS)
BASELINES = ["HPA", "LSTM", "MLP", "Informer", "TimeXer", "iTransformer", "TimeFilter"]

# 判据：训练前写定，写入plan.json后不可修改。比较均为验证集、五种子配对。
CRITERIA = [
    {"claim": "C1_输出参照有效", "better": "TimeXer_LA", "than": "TimeXer", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "C3_外部通路免衰减有效_无LA", "better": "TimeXer_PD", "than": "TimeXer", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "C3_外部通路免衰减有效_有LA", "better": "TimeXer_LA_PD", "than": "TimeXer_LA", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "描述_有C3时PD的收益", "better": "TimeXer_LA_PD_C3Raw", "than": "TimeXer_LA_C3Raw", "min_wins": 4, "mean_lower": True, "fast_not_worse": False},
    {"claim": "C2_控制细节有效_对无标记", "better": "TimeXer_LA_PD_C3Raw", "than": "TimeXer_LA_PD", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "C2_控制细节有效_对全零标记", "better": "TimeXer_LA_PD_C3Raw", "than": "TimeXer_LA_PD_C3Zero", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "C2_保留幅值", "better": "TimeXer_LA_PD_C3Raw", "than": "TimeXer_LA_PD_C3Norm", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
    {"claim": "描述_无PD时C3Raw整体收益", "better": "TimeXer_LA_C3Raw", "than": "TimeXer_LA", "min_wins": 4, "mean_lower": True, "fast_not_worse": True},
]
MAIN_RULE = ("V0始终合格；V1需C3a，V2需C1，V3需C1与C3b并优于V0，V4需V3合格与两项C2并优于V0；"
             "在合格集合中按五种子平均验证RMSE最小选择，1e-12 m内并列优先组件较少者再按编号；"
             "无合格改进退回V0。幅值及交互判据决定机制表述。本轮不启用门控。")
# PD与C3的交互（计划§3.2）：每种子 gain_PD=MSE(V3)−MSE(V4)，gain_wd=MSE(V2)−MSE(V7)，I=gain_PD−gain_wd。
# 只有I均值>0、至少4/5种子I>0、且分块bootstrap区间下界>0时，才写"PD增强了C3Raw的收益"；不写"必要条件"。
INTERACTION = {"pd_without_c3": "TimeXer_LA_PD", "pd_with_c3": "TimeXer_LA_PD_C3Raw",
               "wd_without_c3": "TimeXer_LA", "wd_with_c3": "TimeXer_LA_C3Raw", "min_positive_seeds": 4}
BOOTSTRAP = {"block_samples": 48, "repeats": 2000, "seed": 20261029, "level": 0.95}
PRIMARY_VARIANT = "TimeXer_LA_PD_C3Raw"
# 当前用户要求先实现现有数据上的主线改进，门控及其启动诊断不混入本阶段。
D8_RULE = "本轮不启用D8/H1；后续须按v2.1最近3步本站流量/开度的完整样本四格重新诊断"


# ---------------------------------------------------------------- 版本与路径
def signature_parts():
    """训练结果的版本指纹：原工程代码、改进组件源码、变体定义。判据和本入口脚本不进入指纹，
    修改报告代码不会使已完成的训练失效；但plan.json一经写入，判据不可再改（见plan()）。"""
    # 训练适配器也进入版本校验；仅改变报告排版不会改变已训练权重的身份。
    adapter_source = "\n".join(inspect.getsource(f) for f in (make_adam, worker))
    return {"v1_fingerprint": base.fingerprint(), "variants_source": base.sha256(HERE / "timexer_variants.py"),
            "training_adapter_sha256": hashlib.sha256(adapter_source.encode()).hexdigest(),
            "variants": tv.VARIANTS, "external_path_keys": list(tv.EXTERNAL_PATH_KEYS),
            "detail_patch_len": tv.DETAIL_PATCH_LEN}


def signature():
    return hashlib.sha256(json.dumps(signature_parts(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def context(device=None):
    """读取原配置和数据包；确认原工程代码与v1训练时完全一致，否则停止。"""
    cfg = dict(base.configuration())
    if device:
        cfg["device"] = device
    audit = base.prepare()  # 数据或协议有任何变化时原函数会直接报错
    if base.fingerprint() != V1_RECEIPT["fingerprint"]:
        raise RuntimeError("原工程代码与v1训练时不一致，停止。请勿修改02_训练流程下的文件。")
    return cfg, audit


def run_dir(job):
    return OUT / "training/01_对比实验/runs" / f"{job['task']}_{job['model']}_lr{job['lr']:g}_s{job['seed']}"


def v1_dir(model, seed):
    lr = V1_RECEIPT["selected"][f"{TASK}/{model}"]["lr"]
    return V1_RESULTS / "01_对比实验/runs" / f"{TASK}_{model}_lr{lr:g}_s{seed}"


def done(job):
    dest = run_dir(job)
    if not (dest / "result.json").exists():
        return None
    result = base.read(dest / "result.json")
    if result["fingerprint"] != signature() or base.sha256(dest / "best.pt") != result["checkpoint_sha256"]:
        raise RuntimeError(f"训练记录与当前代码版本不一致：{dest}")
    return result


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 阶段：plan
def plan():
    cfg, _ = context()
    train = base.load(TASK, "train")
    thresholds = np.quantile(np.abs(train["y"] - train["persistence"]), [.5, .9]).tolist()
    record = {"created_at": base.now(), "signature_parts": signature_parts(), "signature": signature(),
              "variants": {name: {"label": LABELS[name], **tv.VARIANTS[name]} for name in VARIANT_NAMES},
              "baselines_reused_from_v1": BASELINES, "criteria": CRITERIA, "main_rule": MAIN_RULE,
              "interaction": INTERACTION, "bootstrap": BOOTSTRAP, "d8_rule": D8_RULE,
              "budget": {"learning_rates": cfg["learning_rates"], "seeds": cfg["seeds"],
                         "rule": "种子42试全部学习率，按验证RMSE选定，再训练其余4个种子",
                         "jobs": len(VARIANT_NAMES) * (len(cfg["learning_rates"]) + len(cfg["seeds"]) - 1)},
              "unchanged": ["Dataset与数据包", "70/10/20划分", "训练段标准化", "MSE损失", "Adam", "weight_decay=1e-4（PD组除外）",
                            "batch", "dropout", "epochs", "patience", "梯度裁剪", "TimeXer主干全部参数"],
              "group_thresholds_m": thresholds, "group_threshold_source": "训练标签一步变化的50%/90%分位数，仅用于评分",
              "test_evaluated": False, "candidate_list_closed": True}
    path = OUT / "plan.json"
    if path.exists():
        old = base.read(path)
        for key in ("criteria", "main_rule", "interaction", "variants", "group_thresholds_m", "d8_rule", "bootstrap"):
            if old[key] != record[key]:
                raise RuntimeError(f"plan.json中的{key}已写定，不能修改。如确需修改，请另建新实验目录。")
        if old["signature"] != record["signature"]:
            raise RuntimeError("改进组件源码在写定计划后被修改，已训练结果将不可复用；请另建新实验目录。")
        print("plan.json 已存在且一致", flush=True)
        return old
    OUT.mkdir(parents=True, exist_ok=True)
    base.write_json(path, record)
    print(f"已写定 {path}", flush=True)
    return record

# ---------------------------------------------------------------- 阶段：selftest
# 旧轮次权重 → 用新代码中结构相同的变体严格加载，验证集逐样本预测必须与旧记录一致。
# 推理时权重衰减不起作用，所以08轮（PD训练）权重用TimeXer_LA_PD加载，11轮（wd=0训练）用TimeXer_LA_C3Raw加载。
R = V1_RESULTS


def _sel_lr(path, default=None):
    return base.read(path)["lr"] if path.exists() else default


REPRODUCTION = [
    {"variant": "TimeXer_LA", "source": "06轮 LastAnchor（wd=1e-4）",
     "weight": lambda s: R / f"06_本地原因验证/experiment3/01_对比实验/runs/A_TimeXer_LastAnchor_lr0.001_s{s}/best.pt",
     "pred": lambda s: R / f"06_本地原因验证/experiment3/01_对比实验/runs/A_TimeXer_LastAnchor_lr0.001_s{s}/val_predictions.csv"},
    {"variant": "TimeXer_LA_PD", "source": "08轮 LastAnchor_PathDecay",
     "weight": lambda s: R / f"08_选择性衰减定位/training/01_对比实验/runs/A_TimeXer_LastAnchor_PathDecay_lr{_sel_lr(R / '08_选择性衰减定位/selection.json'):g}_s{s}/best.pt",
     "pred": lambda s: R / f"08_选择性衰减定位/predictions/s{s}/val.csv"},
    {"variant": "TimeXer_LA_C3Raw", "source": "11轮 LastAnchor_Control3Raw（wd=0，仅核对结构）",
     "weight": lambda s: R / f"11_精简与线性对照/training/01_对比实验/runs/A_TimeXer_LastAnchor_Control3Raw_lr{_sel_lr(R / '11_精简与线性对照/selection_TimeXer_LastAnchor_Control3Raw.json'):g}_s{s}/best.pt",
     "pred": lambda s: R / f"11_精简与线性对照/predictions/TimeXer_LastAnchor_Control3Raw_s{s}/val.csv"},
    {"variant": "TimeXer_LA_PD_C3Zero", "source": "11轮 LastAnchor_Control3Zero（wd=0，仅核对结构）",
     "weight": lambda s: R / f"11_精简与线性对照/training/01_对比实验/runs/A_TimeXer_LastAnchor_Control3Zero_lr{_sel_lr(R / '11_精简与线性对照/selection_TimeXer_LastAnchor_Control3Zero.json'):g}_s{s}/best.pt",
     "pred": lambda s: R / f"11_精简与线性对照/predictions/TimeXer_LastAnchor_Control3Zero_s{s}/val.csv"},
]


def selftest(device=None):
    import forward_test
    forward_test.main()  # R1、R2、R3及报错检查，只用随机张量
    cfg, audit = context(device)
    meta = audit["tasks"][TASK]
    val = base.load(TASK, "val")
    mean, std = base.scale(TASK)
    dev = base.setup(cfg, 42)
    vx = torch.tensor(val["x"], device=dev)
    record = {"baseline_reuse": [], "reproduction": [], "mini_training": [], "tolerance_m": 1e-6}

    # 基线复用：哈希一致；v1代码推理验证集，RMSE与v1记录一致；保存逐样本验证预测供分组评分
    for model_name in BASELINES:
        for seed in cfg["seeds"]:
            dest = v1_dir(model_name, seed)
            result = base.read(dest / "result.json")
            if base.sha256(dest / "best.pt") != result["checkpoint_sha256"]:
                raise RuntimeError(f"v1权重哈希不一致：{dest}")
            base.setup(cfg, seed)
            model = base.build(model_name, meta, cfg, dev)
            model.load_state_dict(torch.load(dest / "best.pt", map_location=dev, weights_only=True))
            pred = base.predict(model, vx, cfg["batch_size"]) * std + mean
            rmse = base.metrics(pred, val["y"], val["persistence"])["RMSE"]
            gap = abs(rmse - result["best_val_RMSE"])
            if gap > 1e-7:  # 06轮在本机复现35个基线时，验证RMSE差均在1e-9 m以下
                raise RuntimeError(f"{model_name} s{seed} 验证RMSE {rmse} 与v1记录 {result['best_val_RMSE']} 不一致")
            pd.DataFrame({"time": val["time"], "observed": val["y"], "predicted": pred, "persistence": val["persistence"]}
                         ).to_csv(_ensure(OUT / "baseline_val" / f"{model_name}_s{seed}.csv"), index=False, encoding="utf-8-sig")
            record["baseline_reuse"].append({"model": model_name, "seed": seed, "val_RMSE": rmse, "abs_gap_m": gap})
            if model_name == "TimeXer":
                # 关闭全部新组件的变体必须与原TimeXer逐样本一致（R2）
                plain = tv.build_variant("TimeXer_PD", meta, cfg, dev)
                plain.load_state_dict(torch.load(dest / "best.pt", map_location=dev, weights_only=True), strict=True)
                diff = float(np.abs(base.predict(plain, vx, cfg["batch_size"]) * std + mean - pred).max())
                if diff > 1e-6:
                    raise RuntimeError(f"关闭新组件后与原TimeXer不一致，最大差 {diff} m")
                record["reproduction"].append({"variant": "TimeXer_PD(组件全关)", "source": "v1 TimeXer", "seed": seed, "max_abs_diff_m": diff})
            del model
    print("基线复用检查通过：35个权重哈希与验证RMSE均与v1一致", flush=True)

    # 旧轮次权重复现（R4）
    for item in REPRODUCTION:
        for seed in cfg["seeds"]:
            weight, pred_file = item["weight"](seed), item["pred"](seed)
            if not weight.exists() or not pred_file.exists():
                record["reproduction"].append({"variant": item["variant"], "source": item["source"], "seed": seed, "skipped": "旧文件不存在"})
                continue
            model = tv.build_variant(item["variant"], meta, cfg, dev)
            model.load_state_dict(torch.load(weight, map_location=dev, weights_only=True), strict=True)
            pred = base.predict(model, vx, cfg["batch_size"]) * std + mean
            old = pd.read_csv(pred_file)["predicted"].to_numpy()
            diff = float(np.abs(pred - old).max())
            if diff > 1e-6:
                raise RuntimeError(f"{item['variant']} 复现 {item['source']} s{seed} 失败，最大差 {diff} m")
            record["reproduction"].append({"variant": item["variant"], "source": item["source"], "seed": seed, "max_abs_diff_m": diff})
            # 正则化不同但结构相同的11轮权重只检查接口，不混进本轮同条件参照。
            if item["variant"] in ("TimeXer_LA", "TimeXer_LA_PD"):
                pd.DataFrame({"time": val["time"], "observed": val["y"], "predicted": pred,
                              "persistence": val["persistence"]}).to_csv(
                    _ensure(OUT / "reference_val" / f"{item['variant']}_s{seed}.csv"),
                    index=False, encoding="utf-8-sig")
            del model
    print("旧轮次权重复现检查通过", flush=True)

    # 小规模训练（R5）：256条样本、2轮，参数分组生效、预测改变、保存后严格重载一致
    record["mini_training"] = mini_training(cfg, meta, dev, mean, std)
    record.update({"status": "passed", "checked_at": base.now(), "signature": signature(), "test_evaluated": False})
    base.write_json(OUT / "selftest.json", record)
    print(f"自检全部通过，记录见 {OUT / 'selftest.json'}", flush=True)


def _ensure(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def make_adam(model, parameters, lr, weight_decay, original_adam, record_path=None, **kwargs):
    """构造Adam。变体声明了参数分组时用分组，否则与原代码完全相同。
    这里只决定参数怎样分组，优化器类型、学习率、衰减系数都取原train.py传入的值。"""
    parameters = list(parameters)
    groups = model.optimizer_groups(weight_decay) if hasattr(model, "optimizer_groups") else None
    if groups is None:
        record = {"groups": 1, "weight_decay": float(weight_decay), "parameters": int(sum(p.numel() for p in parameters))}
        if record_path:
            base.write_json(record_path, record)
        return original_adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    params, record = groups
    grouped = [id(p) for g in params for p in g["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != {id(p) for p in parameters}:
        raise RuntimeError("参数分组与train.py传入的参数不一致（重复或遗漏）")
    if record_path:
        base.write_json(record_path, record)
    return original_adam(params, lr=lr, weight_decay=weight_decay, **kwargs)


def mini_training(cfg, meta, dev, mean, std):
    train = base.load(TASK, "train", 256)
    x = torch.tensor(train["x"], device=dev)
    y = torch.tensor((train["y"] - mean) / std, dtype=torch.float32, device=dev)[:, None]
    rows = []
    for name in VARIANT_NAMES:
        base.setup(cfg, 42)
        model = tv.build_variant(name, meta, cfg, dev)
        before = base.predict(model, x, cfg["batch_size"])
        optimizer = make_adam(model, model.parameters(), 1e-3, cfg["weight_decay"], torch.optim.Adam)
        decays = sorted({g["weight_decay"] for g in optimizer.param_groups})
        model.train()
        for _ in range(2):
            for start in range(0, len(x), 64):
                optimizer.zero_grad(set_to_none=True)
                loss = base.training_loss(model, model(x[start:start + 64]), y[start:start + 64], cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{name} 小规模训练损失非有限")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
        after = base.predict(model, x, cfg["batch_size"])
        if np.array_equal(before, after):
            raise RuntimeError(f"{name} 训练没有改变预测")
        path = _ensure(OUT / "selftest_weights" / f"{name}.pt")
        torch.save(model.state_dict(), path)
        again = tv.build_variant(name, meta, cfg, dev)
        again.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
        np.testing.assert_allclose(base.predict(again, x, cfg["batch_size"]), after, rtol=1e-6, atol=1e-7)
        expected = [0.0, cfg["weight_decay"]] if tv.VARIANTS[name]["path_decay"] else [cfg["weight_decay"]]
        if decays != sorted(expected):
            raise RuntimeError(f"{name} 参数组衰减为 {decays}，应为 {expected}")
        rows.append({"model": name, "param_group_decays": decays, "prediction_changed": True, "reload_equal": True})
        print(f"MINI TRAIN PASS {name} 参数组衰减 {decays}", flush=True)
        del model, again
    return rows


# ---------------------------------------------------------------- 阶段：fit
def worker(job, device=None):
    """单个训练任务：调用原train.py的train_job，只替换模型构造和Adam参数分组。"""
    cfg, audit = context(device)
    base.RESULTS = OUT / "training"                       # 结果写入本实验目录，不碰v1
    base.RUNTIME = OUT / "workers" / f"{job['model']}_s{job['seed']}_lr{job['lr']:g}"
    holder = {}

    def build(name, metadata, cfg_, device_):
        model = tv.build_variant(name, metadata, cfg_, device_)
        holder["model"] = model
        return model

    original_adam = torch.optim.Adam

    def grouped_adam(parameters, lr, weight_decay=0, **kwargs):
        return make_adam(holder["model"], parameters, lr, weight_decay, original_adam,
                         record_path=run_dir(job) / "optimizer_groups.json", **kwargs)

    base.build = build
    torch.optim.Adam = grouped_adam
    try:
        base.train_job(cfg, audit, job, signature())
    finally:
        torch.optim.Adam = original_adam


def pool(jobs, cfg, device, parallel):
    pending = [j for j in jobs if done(j) is None]
    running, failed = [], []
    total = len(jobs)
    while pending or running:
        while pending and len(running) < parallel and not failed:
            job = pending.pop(0)
            dest = run_dir(job)
            if (dest / "epochs.csv").exists() or (dest / "failure.json").exists():
                raise RuntimeError(f"发现未完成的训练，不自动覆盖：{dest}。确认后手动删除该目录再运行。")
            dest.mkdir(parents=True, exist_ok=True)
            handle = (dest / "console.log").open("w", encoding="utf-8")
            command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__)), "--stage", "worker", "--job", json.dumps(job)]
            if device:
                command += ["--device", device]
            proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            running.append((job, proc, handle, time.monotonic()))
            print(f"TRAIN pid={proc.pid} {job}", flush=True)
        kept = []
        for job, proc, handle, started in running:
            if proc.poll() is None and time.monotonic() - started > cfg["job_timeout_seconds"]:
                proc.kill(); proc.wait()
                base.write_json(run_dir(job) / "failure.json", {"reason": "hard_timeout", "job": job})
            code = proc.poll()
            if code is None:
                kept.append((job, proc, handle, started))
            else:
                handle.close()
                if code or done(job) is None:
                    failed.append(job)
                    base.write_json(run_dir(job) / "failure.json", {"reason": "worker_exit", "code": code, "job": job})
        running = kept
        if failed:
            pending = []
        finished = sum((run_dir(j) / "result.json").exists() for j in jobs)
        base.write_json(OUT / "progress.json", {"updated_at": base.now(), "finished": finished, "total": total,
                                                "running": [j for j, *_ in running], "failed": failed})
        if running:
            time.sleep(5)
    if failed:
        raise RuntimeError(f"训练失败，不自动重试，日志见各运行目录console.log：{failed}")


def fit(device=None, parallel=2, models=None):
    cfg, _ = context(device)
    active = list(models) if models else list(VARIANT_NAMES)
    if not active or len(set(active)) != len(active) or any(n not in VARIANT_NAMES for n in active):
        raise ValueError("训练变体清单为空、重复或包含未知名称")
    if parallel < 1:
        raise ValueError("并行任务数必须大于0")
    if not (OUT / "plan.json").exists() or not (OUT / "selftest.json").exists():
        raise RuntimeError("请先依次运行 --stage plan 与 --stage selftest")
    if base.read(OUT / "selftest.json")["signature"] != signature():
        raise RuntimeError("自检之后改进组件源码被修改，请重新运行 --stage selftest")
    plan()  # 再次核对判据未被修改
    base.write_json(OUT / "active_phase.json", {"started_at": base.now(), "signature": signature(),
        "models": active, "jobs": len(active) * 7, "full_ablation_suite": set(active) == set(VARIANT_NAMES),
        "scope": "现有任务A，先完成当前选定变体，不自动启动其余消融或新站点", "test_evaluated": False})
    first = [{"task": TASK, "model": n, "seed": cfg["seeds"][0], "lr": lr} for n in active for lr in cfg["learning_rates"]]
    pool(first, cfg, device, parallel)
    selected = {}
    for name in active:
        candidates = [done({"task": TASK, "model": name, "seed": cfg["seeds"][0], "lr": lr}) for lr in cfg["learning_rates"]]
        best = min(candidates, key=lambda r: r["best_val_RMSE"])
        selected[name] = best["lr"]
        base.write_json(OUT / f"selection_{name}.json", {"model": name, "lr": best["lr"], "metric": "验证RMSE",
                        "candidates": candidates, "signature": signature()})
    rest = [{"task": TASK, "model": n, "seed": s, "lr": selected[n]} for n in active for s in cfg["seeds"][1:]]
    pool(rest, cfg, device, parallel)
    base.write_json(OUT / "fit_complete.json", {"completed_at": base.now(), "signature": signature(),
                    "selected_lr": selected, "completed_variants": active,
                    "full_ablation_suite_complete": set(active) == set(VARIANT_NAMES),
                    "jobs": len(first) + len(rest), "test_evaluated": False})
    print(f"当前阶段训练完成：{len(active)}个变体、{len(first) + len(rest)}次训练", flush=True)

# ---------------------------------------------------------------- 评分工具
def group_masks(y, persistence, thresholds):
    delta = np.abs(y - persistence)
    return {"all": np.ones(len(y), dtype=bool), "stable": delta <= thresholds[0],
            "moderate": (delta > thresholds[0]) & (delta <= thresholds[1]), "fast": delta > thresholds[1]}


def score(pred, y, persistence, thresholds, **extra):
    rows = []
    for group, mask in group_masks(y, persistence, thresholds).items():
        m = base.metrics(pred[mask], y[mask], persistence[mask])
        rows.append({**extra, "group": group, **m, "RMSE_mm": m["RMSE"] * 1000, "MAE_mm": m["MAE"] * 1000})
    return rows


def selected_lr(name):
    return base.read(OUT / f"selection_{name}.json")["lr"]


def variant_weight(name, seed):
    return run_dir({"task": TASK, "model": name, "seed": seed, "lr": selected_lr(name)}) / "best.pt"


def cross_qk_ratio(model, initial):
    """交叉注意力查询/键权重：训练后范数 / 初始化范数，用来观察衰减是否把外部通路压到0。"""
    trained = [p for n, p in model.named_parameters() if ".cross_attention." in "." + n and ("query" in n or "key" in n)]
    start = [p for n, p in initial.named_parameters() if ".cross_attention." in "." + n and ("query" in n or "key" in n)]
    a = torch.sqrt(sum(p.detach().float().square().sum() for p in trained))
    b = torch.sqrt(sum(p.detach().float().square().sum() for p in start))
    return float(a / b)


# ---------------------------------------------------------------- 阶段：evaluate（只用训练集、验证集）
def evaluate(device=None):
    cfg, audit = context(device)
    receipt = base.read(OUT / "fit_complete.json")
    if receipt["signature"] != signature():
        raise RuntimeError("训练版本与当前代码不一致")
    active = receipt.get("completed_variants", VARIANT_NAMES)
    meta, thresholds = audit["tasks"][TASK], base.read(OUT / "plan.json")["group_thresholds_m"]
    mean, std = base.scale(TASK)
    packs = {s: base.load(TASK, s) for s in ("train", "val")}
    dev = base.setup(cfg, 42)
    tensors = {s: torch.tensor(p["x"], device=dev) for s, p in packs.items()}
    rows, norms = [], []
    val = packs["val"]
    rows += score(val["persistence"], val["y"], val["persistence"], thresholds, model="Persistence", label="-", seed=None, split="val")
    for seed in cfg["seeds"]:  # V0 = v1 TimeXer，复用自检阶段保存的验证预测
        frame = pd.read_csv(OUT / "baseline_val" / f"TimeXer_s{seed}.csv")
        rows += score(frame["predicted"].to_numpy(), val["y"], val["persistence"], thresholds, model="TimeXer", label="V0", seed=seed, split="val")
    # 只加入已严格复现的同条件历史参照，不把wd=0旧候选混入本轮。
    for name in ("TimeXer_LA", "TimeXer_LA_PD"):
        if name in active:
            continue
        reference_files = [OUT / "reference_val" / f"{name}_s{seed}.csv" for seed in cfg["seeds"]]
        if not all(p.exists() for p in reference_files):
            raise RuntimeError(f"同条件参照{name}的五种子预测尚未完整复现")
        for seed, p in zip(cfg["seeds"], reference_files):
            frame = pd.read_csv(p)
            np.testing.assert_allclose(frame["observed"].to_numpy(), val["y"], rtol=0, atol=1e-9)
            rows += score(frame["predicted"].to_numpy(), val["y"], val["persistence"], thresholds,
                          model=name, label=LABELS[name], seed=seed, split="val", source="已验证的历史同条件参照")
    for name in active:
        for seed in cfg["seeds"]:
            base.setup(cfg, seed)
            initial = tv.build_variant(name, meta, cfg, dev)  # 与训练时相同种子的初始化
            model = tv.build_variant(name, meta, cfg, dev)
            model.load_state_dict(torch.load(variant_weight(name, seed), map_location=dev, weights_only=True), strict=True)
            norms.append({"model": name, "label": LABELS[name], "seed": seed, "cross_qk_norm_ratio": cross_qk_ratio(model, initial)})
            for split, pack in packs.items():
                pred = base.predict(model, tensors[split], cfg["batch_size"]) * std + mean
                rows += score(pred, pack["y"], pack["persistence"], thresholds, model=name, label=LABELS[name], seed=seed, split=split)
                pd.DataFrame({"time": pack["time"], "observed": pack["y"], "predicted": pred, "persistence": pack["persistence"]}
                             ).to_csv(_ensure(OUT / "predictions" / f"{name}_s{seed}" / f"{split}.csv"), index=False, encoding="utf-8-sig")
            del model, initial
        print(f"EVAL {LABELS[name]} {name}", flush=True)
    save_csv(OUT / "metrics.csv", rows)
    save_csv(OUT / "cross_attention_norms.csv", norms)
    frame = pd.DataFrame(rows)
    summary = (frame[frame.split == "val"].groupby(["label", "model", "group"], dropna=False)[["RMSE_mm", "MAE_mm"]]
               .agg(["mean", "std", "count"]).round(4))
    summary.to_csv(OUT / "验证集均值.csv", encoding="utf-8-sig")
    base.write_json(OUT / "evaluate_complete.json", {"completed_at": base.now(), "signature": signature(),
        "completed_variants": active, "full_ablation_suite_complete": receipt.get("full_ablation_suite_complete", True),
        "test_evaluated": False})
    write_current_summary(frame, receipt)
    print(f"验证集评估完成，见 {OUT / '验证集均值.csv'}", flush=True)


def write_current_summary(frame, receipt):
    """先交付当前主线结果；未完成完整消融时不生成论文式组件结论。"""
    val = frame[(frame.split == "val") & (frame.group == "all")]
    rows = []
    for name, group in val.groupby("model", sort=False):
        rows.append({"model": name, "seeds": int(group.seed.notna().sum()),
            **{f"{metric}_{stat}": float(getattr(group[metric], stat)())
               for metric in ("MAE", "MAPE", "RMSE", "skill_vs_persistence")
               for stat in ("mean", "std") if stat != "std" or group.seed.notna().sum() > 1}})
    save_csv(OUT / "当前主线对比.csv", rows)
    means = {row["model"]: row["RMSE_mean"] for row in rows}
    lines = ["# 当前数据集TimeXer主线改进结果", "",
        "当前阶段只使用原任务A训练集和验证集；未读取测试集计算新成绩。",
        f"本阶段实际完成{receipt['jobs']}次正式训练；完整消融是否完成：{receipt.get('full_ablation_suite_complete', True)}。", "",
        "| 模型 | 五种子验证RMSE均值±标准差（mm） | 验证MAE（mm） | 技能分数 |",
        "|---|---|---|---|"]
    for row in rows:
        std = row.get("RMSE_std")
        rmse = f"{row['RMSE_mean']*1000:.6f}" + (f" ± {std*1000:.6f}" if std is not None else "（确定性参考）")
        lines.append(f"| {row['model']} | {rmse} | {row['MAE_mean']*1000:.6f} | {row['skill_vs_persistence_mean']:.6f} |")
    if PRIMARY_VARIANT in means:
        lines += ["", "## 主方案与参照的差异", ""]
        current = val[val.model == PRIMARY_VARIANT].set_index("seed")
        for ref in ("TimeXer", "TimeXer_LA", "TimeXer_LA_PD"):
            old = val[val.model == ref].set_index("seed")
            wins = int((current.RMSE.sort_index() < old.RMSE.sort_index()).sum())
            change = (means[PRIMARY_VARIANT] / means[ref] - 1) * 100
            lines.append(f"- 相对{ref}：验证RMSE变化{change:+.4f}%，同种子胜出{wins}/5。")
        lines += ["", "原TimeXer与本方案共享数据、Loss、Adam及训练预算；本方案明确包含外部通路免衰减这一正则化差异。",
            "LA与LA_PD参照来自已完成的同条件历史实验，经本次逐样本复现后复用。",
            "本阶段不据此宣称每个组件独立有效、统计显著或跨站点泛化；完整消融与新站点不属于本次先行训练范围。"]
    (OUT / "当前主线结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- 统计工具
def day_blocks(n, size):
    return [np.arange(s, min(s + size, n)) for s in range(0, n, size)]


def block_bootstrap(stat, n, cfg_boot=BOOTSTRAP):
    """配对移动分块重采样，包含首尾且每次恰有n个样本，不重采样成不同长度。"""
    size = min(n, cfg_boot["block_samples"])
    if n < 1:
        raise ValueError("分块区间需要至少一个有效样本")
    rng = np.random.default_rng(cfg_boot["seed"])
    draws = []
    for _ in range(cfg_boot["repeats"]):
        starts = rng.integers(0, n - size + 1, (n + size - 1) // size)
        indices = (starts[:, None] + np.arange(size)).reshape(-1)[:n]
        draws.append(stat(indices))
    alpha = (1 - cfg_boot["level"]) / 2
    return float(stat(np.arange(n))), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def dm_test(loss_a, loss_b, lag=48):
    """Diebold–Mariano检验（Newey–West HAC方差）。loss为逐样本损失，返回(统计量, 双侧p值)。"""
    from math import erf, sqrt
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, centered = len(d), d - d.mean()
    var = centered @ centered / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    stat = d.mean() / sqrt(max(var, 1e-30) / n)
    return float(stat), float(2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2)))))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def load_predictions(split, seeds):
    """读取验证或测试的逐样本预测：{模型名: [B个种子 × N样本]}。"""
    preds = {}
    for name in VARIANT_NAMES:
        preds[name] = np.stack([pd.read_csv(OUT / "predictions" / f"{name}_s{s}" / f"{split}.csv")["predicted"].to_numpy() for s in seeds])
    if split == "val":
        preds["TimeXer"] = np.stack([pd.read_csv(OUT / "baseline_val" / f"TimeXer_s{s}.csv")["predicted"].to_numpy() for s in seeds])
    else:
        y = base.load(TASK, "test")["y"]
        for name in BASELINES:
            frames = [pd.read_csv(v1_dir(name, s) / "test_predictions.csv") for s in seeds]
            for s, f in zip(seeds, frames):  # 复用v1测试预测前，核对样本顺序与观测值完全一致
                if len(f) != len(y) or not np.allclose(f["observed"].to_numpy(), y, rtol=0, atol=1e-9):
                    raise RuntimeError(f"v1 {name} s{s} 的测试预测与当前测试包样本不一致")
            preds[name] = np.stack([f["predicted"].to_numpy() for f in frames])
    return preds


def mean_seed_rmse(pred, y, idx):
    """先逐种子算RMSE，再对种子取平均（与主表口径一致）。"""
    return float(np.mean(np.sqrt(((pred[:, idx] - y[idx]) ** 2).mean(axis=1))))


# ---------------------------------------------------------------- 阶段：judge（只用验证集）
def judge(device=None):
    cfg, audit = context(device)
    if not base.read(OUT / "evaluate_complete.json").get("full_ablation_suite_complete", True):
        raise RuntimeError("当前仅完成主方案阶段，请查看当前主线结果.md；完整消融补齐后才能执行judge")
    if base.read(OUT / "evaluate_complete.json")["signature"] != signature():
        raise RuntimeError("评估版本与当前代码不一致")
    plan_record = base.read(OUT / "plan.json")
    seeds = cfg["seeds"]
    metrics = pd.read_csv(OUT / "metrics.csv")
    val_rows = metrics[(metrics.split == "val") & metrics.seed.notna()].copy()
    val_rows["seed"] = val_rows["seed"].astype(int)

    def per_seed(model, group="all"):
        sub = val_rows[(val_rows.model == model) & (val_rows.group == group)].set_index("seed")
        if sorted(sub.index) != sorted(seeds):
            raise RuntimeError(f"{model} 的验证结果种子不完整：{sorted(sub.index)}")
        return np.array([sub.loc[s, "RMSE"] for s in seeds])

    results = []
    for rule in plan_record["criteria"]:
        a, b = per_seed(rule["better"]), per_seed(rule["than"])
        fa, fb = per_seed(rule["better"], "fast"), per_seed(rule["than"], "fast")
        wins = int((a < b).sum())
        # 转成Python bool：原write_json不接受numpy类型
        checks = {"配对胜出": bool(wins >= rule["min_wins"]),
                  "均值更低": bool(a.mean() < b.mean()) if rule["mean_lower"] else True,
                  "快速段不变差": bool(fa.mean() <= fb.mean()) if rule["fast_not_worse"] else True}
        results.append({**rule, "wins": wins, "better_mean_mm": a.mean() * 1000, "than_mean_mm": b.mean() * 1000,
                        "change_pct": (a.mean() / b.mean() - 1) * 100, "better_fast_mm": fa.mean() * 1000,
                        "than_fast_mm": fb.mean() * 1000, "checks": checks, "passed": all(checks.values())})
    passed = {r["claim"]: r["passed"] for r in results}
    c1 = passed["C1_输出参照有效"]
    c3a = passed["C3_外部通路免衰减有效_无LA"]
    c3b = passed["C3_外部通路免衰减有效_有LA"]
    c3 = c3a and c3b
    c2 = passed["C2_控制细节有效_对无标记"] and passed["C2_控制细节有效_对全零标记"]
    eligible = ["TimeXer"]
    if c3a:
        eligible.append("TimeXer_PD")
    if c1:
        eligible.append("TimeXer_LA")
    def better_than_original(name):
        return (int((per_seed(name) < per_seed("TimeXer")).sum()) >= 4
                and per_seed(name).mean() < per_seed("TimeXer").mean()
                and per_seed(name, "fast").mean() <= per_seed("TimeXer", "fast").mean())
    if c1 and c3b and better_than_original("TimeXer_LA_PD"):
        eligible.append("TimeXer_LA_PD")
    if "TimeXer_LA_PD" in eligible and c2 and better_than_original(PRIMARY_VARIANT):
        eligible.append(PRIMARY_VARIANT)
    minimum = min(float(per_seed(n).mean()) for n in eligible)
    tied = [n for n in eligible if float(per_seed(n).mean()) <= minimum + 1e-12]
    components = {"TimeXer": 0, "TimeXer_PD": 1, "TimeXer_LA": 1, "TimeXer_LA_PD": 2, PRIMARY_VARIANT: 3}
    main = min(tied, key=lambda n: (components[n], int(LABELS[n][1:])))

    # PD×C3 交互（每种子MSE），以及验证集分块bootstrap区间
    val = base.load(TASK, "val")
    preds = load_predictions("val", seeds)
    k = plan_record["interaction"]

    def interaction(idx):
        mse = {m: ((preds[m][:, idx] - val["y"][idx]) ** 2).mean(axis=1) for m in k.values() if isinstance(m, str)}
        gain_pd = mse[k["pd_without_c3"]] - mse[k["pd_with_c3"]]
        gain_wd = mse[k["wd_without_c3"]] - mse[k["wd_with_c3"]]
        return gain_pd - gain_wd

    per_seed_i = interaction(np.arange(len(val["y"])))
    point, low, high = block_bootstrap(lambda idx: float(interaction(idx).mean()), len(val["y"]), plan_record["bootstrap"])
    inter = {"I_per_seed_m2": per_seed_i.tolist(), "I_mean_m2": point, "ci_low": low, "ci_high": high,
             "positive_seeds": int((per_seed_i > 0).sum()),
             "supported": bool(point > 0 and (per_seed_i > 0).sum() >= k["min_positive_seeds"] and low > 0)}

    diagnostics = {"D7": d7(), "D8": d8(audit["tasks"][TASK], preds["TimeXer_LA_PD_C3Raw"], val, plan_record)}
    decision = {"judged_at": base.now(), "signature": signature(), "criteria_results": results, "C1": c1, "C2": c2, "C3": c3,
                "C3a": c3a, "C3b": c3b, "eligible_candidates": eligible,
                "C2_保留幅值": passed["C2_保留幅值"], "PDxC3_interaction": inter, "main_variant": main,
                "main_rule": plan_record["main_rule"], "diagnostics": diagnostics, "test_evaluated": False}
    base.write_json(OUT / "decision.json", decision)
    write_judge_report(decision)
    print(f"判定完成，主方案：{main}。报告见 {OUT / '验证集判定报告.md'}", flush=True)

def d7():
    """训练集目标水位30分钟变化的滞后1自相关。接近−0.5提示变化中测量噪声/量化成分较大；只作线索。"""
    train = base.load(TASK, "train")
    d = train["y"] - train["persistence"]
    q50 = np.quantile(np.abs(d), .5)
    both_stable = (np.abs(d[1:]) <= q50) & (np.abs(d[:-1]) <= q50)

    def corr(a, b):  # 方差为0时相关系数无定义，记为None（原write_json不接受NaN）
        value = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        return None if not np.isfinite(value) else float(value)

    return {"lag1_autocorr_all": corr(d[1:], d[:-1]),
            "lag1_autocorr_stable_pairs": corr(d[1:][both_stable], d[:-1][both_stable]),
            "stable_pairs": int(both_stable.sum()),
            "note": "相邻样本一阶差分的相关；只用训练集；不作为误差下限的证明"}


def d8(meta, v4_preds, val, plan_record):
    # 本阶段不启动门控；不再用旧版“未来平稳标签+上游条件”误判H1。
    return {"status": "deferred", "cells": {}, "rule": plan_record["d8_rule"],
            "supports_H1": False, "note": "False只表示本轮不启动门控，并非诊断已否定H1"}


def write_judge_report(decision):
    lines = ["# 验证集判定报告（TimeXer改进主线）", "",
             f"生成时间：{decision['judged_at']}。只用训练集和验证集，测试集尚未评估。判据来自 plan.json，训练前写定。", "",
             f"**主方案：{decision['main_variant']}**（规则：{decision['main_rule']}）", "",
             "| 判断 | 改进版 | 对照 | 配对胜出 | 改进版RMSE(mm) | 对照RMSE(mm) | 变化 | 快速段(mm) 改进/对照 | 结论 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in decision["criteria_results"]:
        lines.append(f"| {r['claim']} | {LABELS[r['better']]} {r['better']} | {LABELS[r['than']]} {r['than']} | {r['wins']}/5 | "
                     f"{r['better_mean_mm']:.3f} | {r['than_mean_mm']:.3f} | {r['change_pct']:+.2f}% | "
                     f"{r['better_fast_mm']:.3f}/{r['than_fast_mm']:.3f} | {'成立' if r['passed'] else '不成立'} |")
    i = decision["PDxC3_interaction"]
    lines += ["", f"PD×C3交互量 I（mm²）：均值 {i['I_mean_m2'] * 1e6:.4f}，95%区间 [{i['ci_low'] * 1e6:.4f}, {i['ci_high'] * 1e6:.4f}]，"
              f"{i['positive_seeds']}/5 个种子为正。{'可以写' if i['supported'] else '不能写'}“在当前条件下，PD增强了C3Raw的收益”。", "",
              "## D7 噪声线索", "", "```json", json.dumps(decision["diagnostics"]["D7"], ensure_ascii=False, indent=2), "```", "",
              "## D8 平稳段多余误差分格（V4，验证集）", "", "| 分格 | 样本数 | V4 RMSE(mm) | 持续性RMSE(mm) | 每样本多余误差(mm²) | 多余误差占比 |",
              "|---|---|---|---|---|---|"]
    for cell, v in decision["diagnostics"]["D8"]["cells"].items():
        fmt = lambda z, p=3: "-" if z is None else f"{z:.{p}f}"
        lines.append(f"| {cell} | {v['samples']} | {fmt(v['model_RMSE_mm'])} | {fmt(v['persistence_RMSE_mm'])} | "
                     f"{fmt(v['excess_per_sample_mm2'], 4)} | {fmt(v['excess_share'])} |")
    lines += ["", f"本轮不启动H1：{decision['diagnostics']['D8']['rule']}。该诊断留待后续，不作机制结论。", "",
              "所有比较为同一验证段上的五种子配对，属于开发阶段证据；种子不等于独立站点。"]
    (OUT / "验证集判定报告.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------- 阶段：freeze
def freeze():
    cfg, _ = context()
    decision = base.read(OUT / "decision.json")
    if decision["signature"] != signature():
        raise RuntimeError("判定版本与当前代码不一致")
    path = OUT / "frozen.json"
    if path.exists():
        print("frozen.json 已存在，冻结不可重复执行", flush=True)
        return base.read(path)
    weights = {f"{n}_s{s}": base.sha256(variant_weight(n, s)) for n in VARIANT_NAMES for s in cfg["seeds"]}
    record = {"frozen_at": base.now(), "signature": signature(), "signature_parts": signature_parts(),
              "main_variant": decision["main_variant"], "selected_lr": {n: selected_lr(n) for n in VARIANT_NAMES},
              "files_sha256": {p.name: base.sha256(p) for p in (HERE / "timexer_variants.py", HERE / "run_improvement.py",
                                                                 HERE / "forward_test.py", OUT / "plan.json", OUT / "decision.json")},
              "weights_sha256": weights, "note": "冻结后不再修改结构、不再调参；测试集只评估一次"}
    base.write_json(path, record)
    print(f"已冻结。主方案 {record['main_variant']}。请把 {path} 发给导师或合作者留档，再运行 --stage test", flush=True)
    return record


# ---------------------------------------------------------------- 阶段：test（开发测试集，只评估一次）
def test(device=None):
    cfg, audit = context(device)
    frozen = base.read(OUT / "frozen.json")
    if frozen["signature"] != signature():
        raise RuntimeError("当前代码与冻结时不一致，禁止评估测试集")
    for key, digest in frozen["weights_sha256"].items():
        name, seed = key.rsplit("_s", 1)
        if base.sha256(variant_weight(name, int(seed))) != digest:
            raise RuntimeError(f"冻结后的权重被改动：{key}")
    if (OUT / "test" / "test_complete.json").exists():
        print("开发测试集已经评估过，不重复评估。结果见 test 目录。", flush=True)
        return
    seeds, meta = cfg["seeds"], audit["tasks"][TASK]
    thresholds = base.read(OUT / "plan.json")["group_thresholds_m"]
    test_pack = base.load(TASK, "test")
    mean, std = base.scale(TASK)
    dev = base.setup(cfg, 42)
    tx = torch.tensor(test_pack["x"], device=dev)
    y, persistence = test_pack["y"], test_pack["persistence"]
    for name in VARIANT_NAMES:
        for seed in seeds:
            base.setup(cfg, seed)
            model = tv.build_variant(name, meta, cfg, dev)
            model.load_state_dict(torch.load(variant_weight(name, seed), map_location=dev, weights_only=True), strict=True)
            pred = base.predict(model, tx, cfg["batch_size"]) * std + mean
            pd.DataFrame({"time": test_pack["time"], "observed": y, "predicted": pred, "persistence": persistence}
                         ).to_csv(_ensure(OUT / "predictions" / f"{name}_s{seed}" / "test.csv"), index=False, encoding="utf-8-sig")
            del model
    preds = load_predictions("test", seeds)
    rows = score(persistence, y, persistence, thresholds, model="Persistence", label="-", seed=None)
    for name, stack in preds.items():
        for k, seed in enumerate(seeds):
            rows += score(stack[k], y, persistence, thresholds, model=name, label=LABELS.get(name, "基线"), seed=seed)
    save_csv(OUT / "test" / "test_metrics.csv", rows)
    frame = pd.DataFrame(rows)
    frame.groupby(["label", "model", "group"], dropna=False)[["RMSE_mm", "MAE_mm", "skill_vs_persistence"]].agg(["mean", "std"]).round(4
        ).to_csv(OUT / "test" / "开发测试集均值.csv", encoding="utf-8-sig")

    # 主方案对七个基线与各判据对照：ΔRMSE分块bootstrap区间（先逐种子RMSE再平均）+ DM检验 + Holm校正
    main = frozen["main_variant"]
    rivals = [b for b in BASELINES if b != main] + sorted({r["than"] for r in base.read(OUT / "plan.json")["criteria"]} - {main, "TimeXer"})
    comparisons = []
    for rival in rivals:
        delta = lambda idx, a=preds[main], b=preds[rival]: mean_seed_rmse(a, y, idx) - mean_seed_rmse(b, y, idx)
        point, low, high = block_bootstrap(delta, len(y))
        stat, p = dm_test(((preds[main] - y) ** 2).mean(0), ((preds[rival] - y) ** 2).mean(0))
        comparisons.append({"main": main, "rival": rival, "delta_RMSE_mm": point * 1000, "ci_low_mm": low * 1000,
                            "ci_high_mm": high * 1000, "DM_stat": stat, "p_value": p})
    for row, adjusted in zip(comparisons, holm([c["p_value"] for c in comparisons])):
        row["p_holm"] = adjusted
    save_csv(OUT / "test" / "显著性检验.csv", comparisons)
    base.write_json(OUT / "test" / "test_complete.json", {"completed_at": base.now(), "signature": signature(), "main_variant": main,
                    "comparisons": len(comparisons), "note": "开发测试集：06轮诊断曾使用该时段，结果属探索性"})
    print(f"开发测试集评估完成，结果见 {OUT / 'test'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TimeXer主线改进实验（任务A）")
    parser.add_argument("--stage", required=True,
                        choices=["main", "plan", "selftest", "fit", "worker", "evaluate", "judge", "freeze", "test"])
    parser.add_argument("--job", help="仅供内部worker使用")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="默认使用protocol.json中的设备")
    parser.add_argument("--parallel", type=int, default=2, help="同时训练的任务数")
    parser.add_argument("--models", nargs="+", choices=VARIANT_NAMES,
                        help="本阶段只训练指定变体；例如仅主方案为7次，不自动启动完整49次消融")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "main":
        # 一条命令完成主方案7次正式训练及验证汇总；不触发49次消融或测试集。
        plan()
        check_path = OUT / "selftest.json"
        if not check_path.exists() or base.read(check_path).get("signature") != signature():
            selftest(args.device)
        fit(args.device, args.parallel, [PRIMARY_VARIANT])
        evaluate(args.device)
    elif args.stage == "worker":
        worker(json.loads(args.job), args.device)
    elif args.stage == "plan":
        plan()
    elif args.stage == "selftest":
        selftest(args.device)
    elif args.stage == "fit":
        fit(args.device, args.parallel, args.models)
    elif args.stage == "evaluate":
        evaluate(args.device)
    elif args.stage == "judge":
        judge(args.device)
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "test":
        test(args.device)


if __name__ == "__main__":
    main()
