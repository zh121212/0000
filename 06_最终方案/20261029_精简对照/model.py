"""最终选定模型；只含一个结构，无消融开关。输入为原训练尺度的历史。"""
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

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (12, 17):
            raise ValueError("输入必须为[B,12,17]")
        physical = x[:, :, self.physical_indices]
        calendar = x[:, :, self.calendar_indices]
        values = x[:, :, self.config['detail_indices']]
        recent = values[:, -3:]
        detail = torch.cat([torch.zeros_like(values[:, :-3]), recent - recent.mean(1, keepdim=True)], dim=1)
        marks = torch.cat([calendar, detail], dim=-1)
        prediction = self.network(physical, marks, None, None)[:, -1, -1:]
        return prediction + x[:, -1, -1:] - x[:, :, -1].mean(1, keepdim=True)
