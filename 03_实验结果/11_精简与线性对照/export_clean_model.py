"""导出唯一选定结构，不依赖各轮实验编排脚本；严格复核五个权重的预测。"""
from pathlib import Path
import importlib.util
import json
import shutil
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import local_compact_control_experiment as exp
import numpy as np
import pandas as pd
import torch


def main():
    decision=exp.base.read(OUT/'decision.json')
    assert exp.base.read(OUT/'verification.json')['status']=='passed'
    cfg,audit=exp.context();kind=decision['winner_kind'];winner=decision['winner']
    destination=ROOT/'06_最终方案/20261029_精简对照'
    if destination.exists():
        if not (destination/'provenance.json').exists():raise RuntimeError('最终目录已有其他内容，不覆盖')
        assert exp.base.read(destination/'provenance.json')['experiment_signature']==exp.signature()
    destination.mkdir(parents=True,exist_ok=True)
    meta=audit['tasks']['A'];physical=[i for i in range(17) if i not in meta['time_feature_indices']]
    config={'model':winner,'kind':kind,'task':'A','seq_len':12,'pred_len':1,'total_channels':17,
        'physical_indices':physical,'calendar_indices':meta['time_feature_indices'],'detail_indices':([13,14,15] if kind=='control3' else physical[:-1]),
        'seeds':cfg['seeds'],'training':{**cfg,'weight_decay':0.,'selected_lr':decision['selected_lr']},
        'features':meta['features'],'units':{'water':'m','flow':'m3/s','gate':'mm'},
        'input_format':'[B,12,17]; physical fields use copied training scaler, calendar fields remain0..1',
        'backbone_args':{'task_name':'long_term_forecast','features':'MS','seq_len':12,'pred_len':1,'label_len':6,'use_norm':1,'patch_len':3,
            'enc_in':14,'dec_in':14,'c_out':14,'d_model':64,'d_ff':128,'n_heads':4,'e_layers':2,'d_layers':1,'dropout':.1,
            'factor':3,'activation':'gelu','embed':'timeF','freq':'h','distil':True,'alpha':.1,'top_p':.5,'pos':True,'output_attention':False},
        'validation_RMSE_m':decision['validation_RMSE_m'],'validation_MAE_m':decision['validation_MAE_m'],
        'test_evaluated':False}
    exp.base.write_json(destination/'config.json',config)
    init_extra=''
    if kind=='linear':
        init_extra="""        with torch.random.fork_rng(devices=[]):
            self.linear_residual = nn.Linear(204, 1)
            nn.init.zeros_(self.linear_residual.weight)
            nn.init.zeros_(self.linear_residual.bias)
"""
    if kind in ['control3','detail13']:
        forward="""        values = x[:, :, self.config['detail_indices']]
        recent = values[:, -3:]
        detail = torch.cat([torch.zeros_like(values[:, :-3]), recent - recent.mean(1, keepdim=True)], dim=1)
        marks = torch.cat([calendar, detail], dim=-1)
        prediction = self.network(physical, marks, None, None)[:, -1, -1:]
        return prediction + x[:, -1, -1:] - x[:, :, -1].mean(1, keepdim=True)
"""
    else:
        forward="""        prediction = self.network(physical, calendar, None, None)[:, -1, -1:]
        result = prediction + (x[:, -1, -1:] - x[:, :, -1].mean(1, keepdim=True))
"""
        if kind=='linear':
            forward+="""        centered = x.clone()
        centered[:, :, self.physical_indices] = x[:, :, self.physical_indices] - x[:, -1:, self.physical_indices]
        return result + self.linear_residual(centered.flatten(1))
"""
        else:forward+='        return result\n'
    source='''"""最终选定模型；只含一个结构，无消融开关。输入为原训练尺度的历史。"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "02_训练流程/03_模型设计/vendor/tsl"))
from models.TimeXer import Model as TimeXerBackbone


class ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        self.physical_indices = self.config["physical_indices"]
        self.calendar_indices = self.config["calendar_indices"]
        self.network = TimeXerBackbone(SimpleNamespace(**self.config["backbone_args"]))
'''+init_extra+'''
    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (12, 17):
            raise ValueError("输入必须为[B,12,17]")
        physical = x[:, :, self.physical_indices]
        calendar = x[:, :, self.calendar_indices]
'''+forward
    (destination/'model.py').write_text(source,encoding='utf-8')
    predictor='''"""一键预测：默认当前验证段，使用固定训练标准化参数；输出单位m。"""
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
'''
    (destination/'predict.py').write_text(predictor,encoding='utf-8')
    shutil.copy2(ROOT/'01_数据集/03_划分打包/A/scaler.npz',destination/'scaler.npz')
    weights=destination/'weights';weights.mkdir(exist_ok=True);source_paths={}
    for seed in cfg['seeds']:
        job={'task':'A','model':winner,'seed':seed,'lr':decision['selected_lr']}
        if winner in exp.MODES:source_path=exp.path(job)/'best.pt'
        elif kind=='detail13':source_path=exp.parent.path(job)/'best.pt'
        else:source_path=exp.parent.prior.run_path(job,0)/'best.pt'
        shutil.copy2(source_path,weights/f'seed{seed}.pt')
        assert exp.base.sha256(source_path)==exp.base.sha256(weights/f'seed{seed}.pt')
        source_paths[str(seed)]={'path':str(source_path),'sha256':exp.base.sha256(source_path)}
    spec=importlib.util.spec_from_file_location('selected_clean_model',destination/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    device=exp.base.setup(cfg,42);pack=exp.base.load('A','val');mean,std=exp.base.scale('A');x=torch.tensor(pack['x'],device=device);checks=[]
    for seed in cfg['seeds']:
        model=module.ForecastModel().to(device)
        model.load_state_dict(torch.load(weights/f'seed{seed}.pt',map_location=device,weights_only=True),strict=True)
        assert sum(p.numel() for p in model.parameters())==decision['parameters']
        prediction=exp.base.predict(model,x,64)*std+mean
        if winner in exp.MODES:stored=pd.read_csv(OUT/'predictions'/f'{winner}_s{seed}'/'val.csv').predicted.to_numpy()
        elif kind=='detail13':stored=pd.read_csv(exp.parent.OUT/'predictions'/f'{winner}_s{seed}'/'val.csv').predicted.to_numpy()
        else:stored=pd.read_csv(exp.parent.prior.OUT/'predictions'/f'{winner}_wd0_s{seed}'/'val.csv').predicted.to_numpy()
        difference=float(np.max(abs(prediction-stored)));assert difference<1e-6
        batch1=exp.base.predict(model,x[:16],1)*std+mean
        np.testing.assert_allclose(batch1,prediction[:16],rtol=0,atol=1e-6)
        checks.append({'seed':seed,'samples':len(prediction),'max_export_difference_m':difference,'batch1_consistent':True})
        del model
    exp.base.write_json(destination/'export_verification.json',{'status':'passed','checks':checks,'parameters':decision['parameters'],'test_evaluated':False})
    exp.base.write_json(destination/'provenance.json',{'experiment_signature':exp.signature(),'winner':winner,'kind':kind,'source_weights':source_paths,
        'model_sha256':exp.base.sha256(destination/'model.py'),'predict_sha256':exp.base.sha256(destination/'predict.py'),
        'scaler_sha256':exp.base.sha256(destination/'scaler.npz'),'source_baseline_fingerprint':exp.base.fingerprint(),'test_evaluated':False})
    exp.base.write_json(OUT/'clean_export.json',{'directory':str(destination),'winner':winner,'verified':True})
    print('CLEAN_EXPORT',destination)


if __name__=='__main__':main()
