"""一键预测：默认当前验证段，使用固定训练标准化参数；输出单位m。"""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from model import ForecastModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def prepare_device(requested):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(requested)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT / "01_数据集/03_划分打包/A/val.npz")
    parser.add_argument("--seed", type=int, choices=[42,123,2024,2025,3407], default=42)
    parser.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    parser.add_argument("--raw", action="store_true", help="x中的物理字段尚未标准化时使用；日历仍须按原规则编码")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        values = {k:data[k] for k in data.files}
    x = np.asarray(values["x"], dtype=np.float64)
    if x.ndim == 2: x = x[None]
    if tuple(x.shape[1:]) != (12,17) or not np.isfinite(x).all():
        raise ValueError("x必须为有限数值[B,12,17]或[12,17]")
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    with np.load(HERE / "scaler.npz", allow_pickle=False) as scaler:
        mean, std = float(scaler["y_mean"]), float(scaler["y_std"])
        if args.raw:
            ids = config["physical_indices"]
            x[:,:,ids] = (x[:,:,ids] - scaler["physical_mean"]) / scaler["physical_std"]
    device = prepare_device(args.device)
    model = ForecastModel().to(device)
    model.load_state_dict(torch.load(HERE / "weights" / f"seed{args.seed}.pt", map_location=device, weights_only=True), strict=True)
    model.eval()
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.inference_mode():
        prediction = np.concatenate([model(tensor[i:i+64]).cpu().numpy()[:,0] for i in range(0,len(tensor),64)]).astype(np.float64)*std+mean
    frame = pd.DataFrame({"predicted_water_level_m":prediction})
    if "time" in values: frame.insert(0,"time",values["time"])
    if "y" in values: frame["observed_water_level_m"] = values["y"]
    output = args.output or HERE / f"prediction_seed{args.seed}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    result = {"model":config["model"],"seed":args.seed,"samples":len(frame),"unit":"m","output":str(output)}
    if "y" in values:
        error = prediction - values["y"]
        result.update({"RMSE_m":float(np.sqrt(np.mean(error**2))),"MAE_m":float(np.mean(np.abs(error)))})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
