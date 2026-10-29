"""七个模型读取同一真实历史张量；HPA明确属于历史输入适配。"""
from pathlib import Path
from types import SimpleNamespace
import importlib
import importlib.util
import math
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/tsl"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def positional(length, dim):
    result = torch.zeros(length, dim)
    scales = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
    values = torch.arange(length).float()[:, None] * scales
    result[:, 0::2], result[:, 1::2] = values.sin(), values.cos()
    return result


class HPAHistory(nn.Module):
    def __init__(self, name, metadata, cfg, device):
        super().__init__()
        channels, up_channels = metadata["total_channels"], metadata["up_channels"]
        self.name, self.up_channels = name, up_channels
        self.flow_index = metadata["upstream_flow_index"]
        length, dp = cfg["seq_len"], cfg["dropout"]
        self.keep = [i for i in range(up_channels) if not (name == "HPA_no_time" and i in metadata["time_feature_indices"])]
        self.input_projection = nn.Linear(len(self.keep), 32)
        self.register_buffer("position", positional(length, 32))
        self.position_dropout = nn.Dropout(dp)
        self.register_buffer("mask", torch.ones(length, length).tril())
        self.attention = nn.ModuleList()
        if name != "HPA_no_attention":
            spec = importlib.util.spec_from_file_location("local_original_hpa", HERE / "vendor/hpa/item_lstm_Model.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for _ in range(5):
                layer = module.MultiHeadAttentionLayer(160, 5, dp, device)
                if length != 12:
                    layer.bn = nn.BatchNorm1d(length)
                self.attention.append(layer)
        if name != "HPA_uniform_pool":
            self.flow_embedding = nn.Linear(1, 32)
        # 目标站每项物理变量及每孔开度均读入，分支宽度由实际字段决定。
        self.local_branch = nn.Sequential(nn.Linear(length * metadata["local_channels"], 32), nn.Dropout(dp), nn.Tanh())
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        up, local = x[:, :, :self.up_channels], x[:, :, self.up_channels:]
        h = self.position_dropout(self.input_projection(up[:, :, self.keep]) + self.position)
        for layer in self.attention:
            h, _ = layer(h, h, h, self.mask)
        if self.name == "HPA_uniform_pool":
            pooled = h.sigmoid().mean(dim=1)
        else:
            weights = self.flow_embedding(up[:, :, self.flow_index:self.flow_index+1]).tanh().amax(dim=-1).softmax(dim=1)
            pooled = torch.bmm(weights[:, None], h.sigmoid()).squeeze(1)
        return self.output(torch.cat([pooled, self.local_branch(local.flatten(1))], dim=-1))


class Basic(nn.Module):
    def __init__(self, name, channels, cfg):
        super().__init__()
        self.name = name
        if name == "MLP":
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(cfg["seq_len"]*channels, 64),
                                         nn.Tanh(), nn.Dropout(cfg["dropout"]), nn.Linear(64, 64),
                                         nn.Tanh(), nn.Dropout(cfg["dropout"]), nn.Linear(64, 1))
        else:
            self.encoder = nn.LSTM(channels, 64, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(64, 1))

    def forward(self, x):
        if self.name == "MLP":
            return self.network(x)
        h, _ = self.encoder(x)
        return self.head(h[:, -1])


class NativeForecast(nn.Module):
    def __init__(self, name, channels, cfg, time_feature_indices):
        super().__init__()
        self.name = name
        self.auxiliary_loss = None
        self.label_len = cfg["seq_len"] // 2
        self.calendar_indices = list(time_feature_indices)
        self.physical_indices = [i for i in range(channels) if i not in self.calendar_indices]
        calendar_mask = torch.zeros(1, 1, channels, dtype=torch.bool)
        calendar_mask[:, :, self.calendar_indices] = True
        self.register_buffer("calendar_mask", calendar_mask, persistent=False)
        # 两种倒置嵌入均支持独立时间输入；日历不进入物理变量的窗口归一化。
        core_channels = len(self.physical_indices) if name in ("TimeXer", "iTransformer") else channels
        args = SimpleNamespace(task_name="long_term_forecast", features="MS", seq_len=cfg["seq_len"],
                               pred_len=1, label_len=self.label_len, use_norm=1, patch_len=3,
                               enc_in=core_channels, dec_in=core_channels, c_out=core_channels, d_model=64,
                               d_ff=128, n_heads=4, e_layers=2, d_layers=1, dropout=cfg["dropout"],
                               factor=3, activation="gelu", embed="timeF", freq="h", distil=True,
                               alpha=.1, top_p=.5, pos=True, output_attention=False)
        self.network = importlib.import_module(f"models.{name}").Model(args)

    def forward(self, x):
        self.auxiliary_loss = None
        if self.name == "Informer":
            # 解码器仅复制已知历史；未来槽位严格为零，绝不复制标签。
            decoder = torch.cat([x[:, -self.label_len:], torch.zeros_like(x[:, :1])], dim=1)
            pred = self.network(x, None, decoder, None)
        elif self.name == "TimeFilter":
            # TSL.forecast丢弃了moe_loss；在适配器内保留它，vendor源文件不改。
            net = self.network
            z = net.norm(x, "norm")
            # 日历已编码为0至1，保留其绝对位置；物理通道继续使用原归一化。
            z = torch.where(self.calendar_mask, x, z)
            batch, length, channels = z.shape
            z = net.patch_embed(z.permute(0, 2, 1).reshape(batch, channels*length))
            z, self.auxiliary_loss = net.backbone(z, net._get_mask(x.device), net.alpha)
            pred = net.head(z.reshape(batch, channels, net.num_patches, net.dim).flatten(start_dim=-2))
            pred = net.norm(pred.permute(0, 2, 1), "denorm")
        elif self.name in ("TimeXer", "iTransformer"):
            physical = x[:, :, self.physical_indices]
            calendar = x[:, :, self.calendar_indices]
            # 物理变量中的目标水位仍为最后一列；最终只取它的单步预测。
            pred = self.network(physical, calendar, None, None)
        else:
            pred = self.network(x, None, None, None)
        return pred[:, -1, -1:]


def build(name, metadata, cfg, device):
    channels = metadata["total_channels"]
    if name.startswith("HPA"):
        model = HPAHistory(name, metadata, cfg, device)
    elif name in ("MLP", "LSTM"):
        model = Basic(name, channels, cfg)
    else:
        model = NativeForecast(name, channels, cfg, metadata["time_feature_indices"])
    return model.to(device)


def training_loss(model, pred, labels, cfg):
    mse = (pred-labels).square().mean()
    aux = getattr(model, "auxiliary_loss", None)
    return mse + cfg["timefilter_aux_weight"] * aux if aux is not None else mse
