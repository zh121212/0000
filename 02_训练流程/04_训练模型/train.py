"""统一入口。所有输出位于本目录，不使用另一项条件预测实验的结果。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
FLOW = ROOT / "02_训练流程"
for relative in ("02_数据划分打包", "03_模型设计", "05_evaluate"):
    sys.path.insert(0, str(FLOW / relative))
import numpy as np
import pandas as pd
import torch
from prepare import prepare, configuration, load, PACKS, write_json, sha256
from adapters import build, training_loss
from metrics import metrics
from report import save_unified_results
from plot_predictions import save_prediction_figures

RESULTS = ROOT / "03_实验结果"
RUNTIME = RESULTS / "运行记录"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint():
    sources = [ROOT / "run.py", ROOT / "04_实验配置/protocol.json"]
    sources += sorted(FLOW.rglob("*.py"))
    sources += sorted((ROOT / "01_数据集/01_源文件快照").glob("*.csv"))
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def setup(cfg, seed, override=None):
    device = torch.device(override or cfg["device"])
    torch.set_num_threads(cfg["cpu_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求CUDA，但当前不可用")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def predict(model, x, batch_size):
    model.eval()
    # Informer的稀疏注意力在eval时仍会抽样；固定评估随机流且不改变训练随机流。
    devices = [x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(117)
        if x.is_cuda:
            torch.cuda.manual_seed_all(117)
        result = [model(x[start:start+batch_size]).detach().cpu().numpy()[:, 0]
                  for start in range(0, len(x), batch_size)]
    pred = np.concatenate(result).astype(np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("预测出现非有限数值")
    return pred


def scale(task):
    with np.load(PACKS / task / "scaler.npz", allow_pickle=False) as scaler:
        return float(scaler["y_mean"]), float(scaler["y_std"])


def job_count(cfg):
    return len(cfg["tasks"]) * (len(cfg["models"]) * (len(cfg["learning_rates"])+len(cfg["seeds"])-1)
                                + len(cfg["ablations"]) * len(cfg["seeds"]))


def training_plan(cfg):
    return {"status": "plan_only_no_training", "task": cfg["tasks"], "models": cfg["models"],
            "seeds": cfg["seeds"], "learning_rates": cfg["learning_rates"],
            "candidate_jobs": len(cfg["tasks"])*len(cfg["models"])*len(cfg["learning_rates"]),
            "final_seed_results": len(cfg["tasks"])*len(cfg["models"])*len(cfg["seeds"]),
            "distinct_training_jobs": job_count(cfg), "research_backbone": cfg["research_backbone"],
            "note": "种子42的最优学习率训练同时计入最终五种子，故实际训练49次，最终报告35次。"}


def smoke(cfg, audit, signature, override=None):
    receipt = RESULTS / "04_接口检查/smoke.json"
    if receipt.exists():
        previous = read(receipt)
        if previous.get("fingerprint") == signature and all(row["device"] == (override or cfg["device"]) for row in previous.get("checks", [])):
            return previous
    checks = []
    for task, meta in audit["tasks"].items():
        pack = load(task, "train", 256)
        val = load(task, "val", 64)
        mean, std = scale(task)
        for name in cfg["models"] + cfg["ablations"]:
            device = setup(cfg, 42, override)
            x = torch.tensor(pack["x"], device=device)
            y = torch.tensor((pack["y"]-mean)/std, dtype=torch.float32, device=device)[:, None]
            model = build(name, meta, cfg, device)
            before = predict(model, x, cfg["batch_size"])
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=cfg["weight_decay"])
            model.train()
            losses = []
            for _ in range(2):
                for start in range(0, len(x), 64):
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(x[start:start+64])
                    if pred.shape != y[start:start+64].shape:
                        raise ValueError(f"{name}输出维度错误")
                    loss = training_loss(model, pred, y[start:start+64], cfg)
                    if not torch.isfinite(loss).all():
                        raise ValueError(f"{name}损失无效")
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise ValueError(f"{name}梯度无效")
                    if name == "TimeFilter":
                        gate_grads = [p.grad for k, p in model.named_parameters() if "mask_moe.gate" in k]
                        if not gate_grads or not all(g is not None and g.abs().sum() > 0 for g in gate_grads):
                            raise ValueError("TimeFilter门控模块没有得到辅助损失梯度")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            updated = predict(model, x, cfg["batch_size"])
            if np.array_equal(before, updated):
                raise ValueError(f"{name}训练没有改变预测")
            path = RESULTS / "04_接口检查" / f"{task}_{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            reloaded = build(name, meta, cfg, device)
            reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            np.testing.assert_allclose(predict(reloaded, x, cfg["batch_size"]), updated, rtol=1e-6, atol=1e-7)
            # 确认两个分支的信息确实进入预测；消融有意移除的时间列不在此强制检查。
            # 用固定随机流下的输入扰动检查依赖，兼容原模型的就地归一化。
            probe = x[:8].detach().clone()
            dependence = {}
            if name.startswith("HPA") or name in ("MLP", "LSTM"):
                reloaded.eval()
                probe.requires_grad_(True)
                # cuDNN的RNN不支持eval模式反传；此处仅做输入连接诊断。
                with torch.backends.cudnn.flags(enabled=False):
                    grad = torch.autograd.grad(reloaded(probe).sum(), probe)[0]
                dependence = {"method": "input_gradient", "upstream_gradient_sum": float(grad[:, :, :meta["up_channels"]].abs().sum()),
                              "local_gradient_sum": float(grad[:, :, meta["up_channels"]:].abs().sum())}
                connected = dependence["upstream_gradient_sum"] > 0 and dependence["local_gradient_sum"] > 0
            else:
                original = predict(reloaded, probe, 8)
                up_changed, local_changed = probe.clone(), probe.clone()
                up_changed[:, -1, :meta["up_channels"]] += .5
                local_changed[:, -1, meta["up_channels"]:] += .5
                dependence = {"method": "fixed_rng_perturbation", "upstream_max_prediction_change": float(np.abs(original-predict(reloaded, up_changed, 8)).max()),
                              "local_max_prediction_change": float(np.abs(original-predict(reloaded, local_changed, 8)).max())}
                connected = dependence["upstream_max_prediction_change"] > 0 and dependence["local_max_prediction_change"] > 0
            if not connected:
                raise ValueError(f"{name}没有使用上游或本站历史")
            validation = torch.tensor(val["x"], device=device)
            val_metrics = metrics(predict(reloaded, validation, cfg["batch_size"])*std+mean, val["y"], val["persistence"])
            checks.append({"task": task, "model": name, "device": str(device), "samples": 256, "epochs": 2,
                           "input_shape": list(x.shape), "output_shape": [256, 1], "finite_gradient": True,
                           "both_histories_connected": True, "dependence_check": dependence, "reload_equal": True, "first_loss": losses[0],
                           "last_loss": losses[-1], "validation_subset_metrics": val_metrics})
            write_json(RESULTS / "04_接口检查/progress.json", {"completed": checks, "updated_at": now()})
            print(f"SMOKE PASS {task}/{name}", flush=True)
            del model, reloaded, optimizer, x, y, validation, probe
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = {"status": "passed", "fingerprint": signature, "updated_at": now(), "checks": checks,
              "test_set_evaluated": False, "note": "256条训练样本、2轮，只验证流程，不是正式对比结果"}
    write_json(receipt, result)
    return result


def job_dir(job):
    category = "02_消融实验" if job["model"].startswith("HPA_") else "01_对比实验"
    key = f"{job['task']}_{job['model']}_lr{job['lr']:g}_s{job['seed']}"
    return RESULTS / category / "runs" / key


def complete(job, signature):
    dest = job_dir(job)
    if not (dest / "result.json").exists():
        return None
    value = read(dest / "result.json")
    if value["fingerprint"] != signature or sha256(dest / "best.pt") != value["checkpoint_sha256"]:
        raise ValueError(f"实验版本或权重改变：{dest}")
    return value


def train_job(cfg, audit, job, signature):
    dest = job_dir(job)
    if complete(job, signature):
        return
    if (dest / "epochs.csv").exists() or (dest / "failure.json").exists():
        raise RuntimeError(f"发现未完成实验，不自动覆盖重跑：{dest}")
    dest.mkdir(parents=True, exist_ok=True)
    device = setup(cfg, job["seed"])
    train, val = load(job["task"], "train"), load(job["task"], "val")
    mean, std = scale(job["task"])
    x = torch.tensor(train["x"], device=device)
    y = torch.tensor((train["y"]-mean)/std, dtype=torch.float32, device=device)[:, None]
    vx = torch.tensor(val["x"], device=device)
    model = build(job["model"], audit["tasks"][job["task"]], cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=job["lr"], weight_decay=cfg["weight_decay"])
    best, best_epoch, stale, started = float("inf"), 0, 0, time.monotonic()
    write_json(dest / "job.json", {**job, "fingerprint": signature, "started_at": now()})
    with (dest / "epochs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_objective", "val_MAE", "val_RMSE", "seconds"])
        writer.writeheader()
        for epoch in range(1, cfg["epochs"]+1):
            model.train()
            order = torch.randperm(len(y), device=device)
            loss_sum = 0.
            for start in range(0, len(y), cfg["batch_size"]):
                ids = order[start:start+cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model, model(x[ids]), y[ids], cfg)
                if not torch.isfinite(loss).all():
                    raise ValueError("训练损失非有限")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(ids)
            scores = metrics(predict(model, vx, cfg["batch_size"])*std+mean, val["y"], val["persistence"])
            if scores["RMSE"] < best:
                best, best_epoch, stale = scores["RMSE"], epoch, 0
                torch.save(model.state_dict(), dest / "best.pt")
            else:
                stale += 1
            writer.writerow({"epoch": epoch, "train_objective": loss_sum/len(y), "val_MAE": scores["MAE"],
                             "val_RMSE": scores["RMSE"], "seconds": time.monotonic()-started})
            handle.flush()
            write_json(RUNTIME / "worker.json", {"updated_at": now(), "job": job, "epoch": epoch,
                                                "best_epoch": best_epoch, "best_val_RMSE": best,
                                                "elapsed_seconds": time.monotonic()-started})
            if stale >= cfg["patience"]:
                break
    result = {**job, "status": "completed", "fingerprint": signature, "best_val_RMSE": best,
              "best_epoch": best_epoch, "epochs_run": epoch, "training_seconds": time.monotonic()-started,
              "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
              "checkpoint_sha256": sha256(dest / "best.pt"), "test_set_evaluated": False}
    write_json(dest / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run_child(cfg, audit, job, signature, counts):
    if complete(job, signature):
        return
    dest = job_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME / "status.json", {"status": "running", "pid": os.getpid(), "updated_at": now(),
                                        "job": job, "completed_now": counts[0], "total_jobs": job_count(cfg)})
    print(f"TRAIN {job}", flush=True)
    with (dest / "console.log").open("w", encoding="utf-8") as output:
        worker = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--stage", "job", "--job", json.dumps(job)],
                                  stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            code = worker.wait(timeout=cfg["job_timeout_seconds"])
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
            write_json(dest / "failure.json", {"reason": "hard_timeout", "job": job, "updated_at": now()})
            raise RuntimeError(f"单项实验超时：{job}")
        except KeyboardInterrupt:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            write_json(dest / "interrupted.json", {"reason": "user_interrupt", "job": job, "updated_at": now()})
            raise
    if code != 0:
        write_json(dest / "failure.json", {"reason": "worker_exit", "code": code, "job": job, "updated_at": now()})
        raise RuntimeError(f"实验失败，队列停止；见{dest / 'console.log'}")
    if not complete(job, signature):
        raise RuntimeError("进程退出但没有完整训练结果")
    counts[0] += 1


def fit_all(cfg, audit, signature):
    selected, counts = {}, [0]
    for task in cfg["tasks"]:
        for name in cfg["models"]:
            candidates = []
            for lr in cfg["learning_rates"]:
                job = {"task": task, "model": name, "lr": lr, "seed": cfg["seeds"][0]}
                run_child(cfg, audit, job, signature, counts)
                candidates.append(complete(job, signature))
            best = min(candidates, key=lambda row: row["best_val_RMSE"])
            selected[f"{task}/{name}"] = {"task": task, "model": name, "lr": best["lr"],
                                           "selection_seed": cfg["seeds"][0], "best_val_RMSE": best["best_val_RMSE"]}
            write_json(RUNTIME / "selection.json", {"fingerprint": signature, "models": selected})
    for task in cfg["tasks"]:
        for name in cfg["models"]:
            lr = selected[f"{task}/{name}"]["lr"]
            for seed in cfg["seeds"][1:]:
                run_child(cfg, audit, {"task": task, "model": name, "lr": lr, "seed": seed}, signature, counts)
        # 消融继承完整HPA在同一渠段的学习率，不使用测试集选消融参数。
        for name in cfg["ablations"]:
            lr = selected[f"{task}/HPA"]["lr"]
            for seed in cfg["seeds"]:
                run_child(cfg, audit, {"task": task, "model": name, "lr": lr, "seed": seed}, signature, counts)
    write_json(RUNTIME / "fit_complete.json", {"fingerprint": signature, "completed_at": now(), "selected": selected})


def evaluate(cfg, audit, signature):
    receipt = read(RUNTIME / "fit_complete.json")
    if receipt["fingerprint"] != signature:
        raise ValueError("训练版本与评估版本不一致")
    comparison, ablations = [], []
    for task, meta in audit["tasks"].items():
        test = load(task, "test")
        mean, std = scale(task)
        baseline = {"task": task, "model": "Persistence", "seed": None, **metrics(test["persistence"], test["y"], test["persistence"]),
                    "training_seconds": 0., "parameters": 0}
        comparison.append(baseline)
        for name in cfg["models"] + cfg["ablations"]:
            lr = receipt["selected"][f"{task}/{'HPA' if name.startswith('HPA_') else name}"]["lr"]
            for seed in cfg["seeds"]:
                job = {"task": task, "model": name, "lr": lr, "seed": seed}
                result = complete(job, signature)
                if result is None:
                    raise ValueError(f"未完成训练：{job}")
                device = setup(cfg, seed)
                model = build(name, meta, cfg, device)
                model.load_state_dict(torch.load(job_dir(job)/"best.pt", map_location=device, weights_only=True))
                x = torch.tensor(test["x"], device=device)
                started = time.monotonic()
                pred = predict(model, x, cfg["batch_size"])*std+mean
                scores = metrics(pred, test["y"], test["persistence"])
                row = {**job, **scores, "best_epoch": result["best_epoch"], "training_seconds": result["training_seconds"],
                       "parameters": result["parameters"], "inference_seconds": time.monotonic()-started}
                write_json(job_dir(job)/"test_metrics.json", {**row, "fingerprint": signature})
                pd.DataFrame({"time": test["time"], "target_index": test["target_index"], "observed": test["y"],
                              "predicted": pred, "persistence": test["persistence"]}).to_csv(job_dir(job)/"test_predictions.csv", index=False, encoding="utf-8-sig")
                (ablations if name.startswith("HPA_") else comparison).append(row)
                del model, x
                torch.cuda.empty_cache()
    for category, rows in [("01_对比实验", comparison), ("02_消融实验", ablations)]:
        if not rows:
            continue
        (RESULTS/category).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS/category/"all_seed_metrics.csv", index=False, encoding="utf-8-sig")
        summary = df.groupby(["task", "model"], sort=False)[["MAE", "MAPE", "RMSE", "skill_vs_persistence"]].agg(["mean", "std", "count"])
        summary.to_csv(RESULTS/category/"summary.csv", encoding="utf-8-sig")
    save_unified_results(comparison, cfg, RESULTS/"01_对比实验/统一结果表.csv")
    plot_all(cfg, signature)
    write_json(RUNTIME/"status.json", {"status": "completed", "updated_at": now(), "fingerprint": signature,
                                       "test_used_for_selection": False, "formal_training_jobs": job_count(cfg)})


def plot_all(cfg, signature):
    receipt = read(RUNTIME / "fit_complete.json")
    if receipt["fingerprint"] != signature:
        raise ValueError("训练版本与作图版本不一致")
    for task in cfg["tasks"]:
        files = {}
        for name in cfg["models"]:
            lr = receipt["selected"][f"{task}/{name}"]["lr"]
            files[name] = {}
            for seed in cfg["seeds"]:
                job = {"task": task, "model": name, "lr": lr, "seed": seed}
                record = read(job_dir(job) / "test_metrics.json")
                if record["fingerprint"] != signature:
                    raise ValueError("测试预测与作图版本不一致")
                files[name][seed] = job_dir(job) / "test_predictions.csv"
        save_prediction_figures(cfg, task, files, RESULTS / "05_预测对比图" / task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "audit", "prepare", "smoke", "all", "evaluate", "plot", "status", "job"], default="status")
    parser.add_argument("--job")
    parser.add_argument("--smoke-device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    cfg = configuration()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        print(json.dumps(training_plan(cfg), ensure_ascii=False, indent=2))
        return
    if args.stage == "audit":
        subprocess.run([sys.executable, str(FLOW/"01_数据处理/audit_data.py")], check=True)
        return
    if args.stage == "status":
        print(json.dumps(read(RUNTIME/"status.json") if (RUNTIME/"status.json").exists() else {"status": "not_started"}, ensure_ascii=False, indent=2))
        return
    audit = prepare()
    if args.stage == "prepare":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    signature = fingerprint()
    if args.stage == "plot":
        plot_all(cfg, signature)
        return
    if args.stage == "job":
        train_job(cfg, audit, json.loads(args.job), signature)
        return
    if args.stage in ("smoke", "all"):
        packages = {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "einops", "reformer-pytorch"]}
        write_json(RUNTIME/"environment.json", {"recorded_at": now(), "python": sys.executable, "python_version": sys.version,
                                               "packages": packages, "cuda_build": torch.version.cuda,
                                               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                               "fingerprint": signature})
        smoke(cfg, audit, signature, args.smoke_device)
    if args.stage == "all":
        fit_all(cfg, audit, signature)
    if args.stage in ("all", "evaluate"):
        evaluate(cfg, audit, signature)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        write_json(RUNTIME/"status.json", {"status": "paused_by_user", "updated_at": now(), "note": "训练已停止；不自动恢复，未完成运行不会冒充完整结果"})
        raise SystemExit(130)
    except Exception as error:
        write_json(RUNTIME/"status.json", {"status": "failed", "updated_at": now(), "error": str(error)})
        raise
