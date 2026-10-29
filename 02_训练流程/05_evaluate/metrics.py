"""只在原始水位尺度计算指标；相同时间的保持不变预测作为基线。"""
import numpy as np


def metrics(pred, true, persistence):
    pred, true, persistence = [np.asarray(a, dtype=np.float64) for a in (pred, true, persistence)]
    if pred.shape != true.shape or true.shape != persistence.shape or not all(np.isfinite(a).all() for a in (pred, true, persistence)):
        raise ValueError("预测形状或数值无效")
    err, base = pred-true, persistence-true
    mse, baseline_mse = np.mean(err**2), np.mean(base**2)
    return {"MAE": float(np.abs(err).mean()), "MAPE": float(np.mean(np.abs(err / true)) * 100) if np.all(true != 0) else None,
            "RMSE": float(np.sqrt(mse)),
            "skill_vs_persistence": float(1-mse/baseline_mse) if baseline_mse > 0 else None,
            "n": len(true)}
