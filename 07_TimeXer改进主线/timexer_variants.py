"""TimeXer主线改进的独立组件。

本文件只新增模块，不修改原工程任何文件：
- 原TimeXer主干（vendor/tsl/models/TimeXer.py）原样调用；
- 原适配器 adapters.NativeForecast 作为父类，参数名、参数形状与v1完全一致，
  因此06、08、11轮保存的旧权重可以直接严格加载，用于复现检查；
- 新组件都不含可训练参数，也不注册buffer，不改变state_dict。

三个组件分别对应三个研究问题：
1. LastAnchorOutput      —— 输出参照：窗口均值参照改为最后实测水位（问题P3）；
2. ControlDetailTokens   —— 短时控制动态：目标站流量与各孔开度最近3步的细节标记（问题P4）；
3. external_path_groups  —— 外部通路免衰减：外部嵌入与交叉注意力不受权重衰减（问题P2）。
"""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "02_训练流程/03_模型设计"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from adapters import NativeForecast  # noqa: E402  adapters会自行加入vendor路径

# 外部信息通路的参数名特征：外部变量嵌入、每层的交叉注意力
EXTERNAL_PATH_KEYS = (".ex_embedding.", ".cross_attention.")
# 控制细节只看最近一个分块（与TimeXer的patch_len=3一致）
DETAIL_PATCH_LEN = 3

# 变体名称 → 三个开关。名称只能从这里选，避免拼写错误悄悄生成别的模型。
VARIANTS = {
    "TimeXer_PD":             {"anchor": False, "control": None,   "path_decay": True},   # V1
    "TimeXer_LA":             {"anchor": True,  "control": None,   "path_decay": False},  # V2
    "TimeXer_LA_PD":          {"anchor": True,  "control": None,   "path_decay": True},   # V3
    "TimeXer_LA_PD_C3Raw":    {"anchor": True,  "control": "raw",  "path_decay": True},   # V4 主方案候选
    "TimeXer_LA_PD_C3Zero":   {"anchor": True,  "control": "zero", "path_decay": True},   # V5 全零对照
    "TimeXer_LA_PD_C3Norm":   {"anchor": True,  "control": "norm", "path_decay": True},   # V6 窗口归一化对照
    "TimeXer_LA_C3Raw":       {"anchor": True,  "control": "raw",  "path_decay": False},  # V7 外部通路照常衰减
}


def _shape_error(where, expected, tensor):
    raise ValueError(f"[{where}] 张量形状不符：期望 {expected}，实际 {tuple(tensor.shape)}")


class LastAnchorOutput(nn.Module):
    """输出参照修正（无参数）。

    原TimeXer：预测 = 窗口均值 + 窗口标准差 × 头部输出。
    本模块把参照从窗口均值换成最后实测值：预测 = 原预测 − 窗口均值 + 最后实测值。
    头部输出为0时，预测就等于"水位保持不变"。全部计算在训练段标准化坐标中完成。
    """

    def forward(self, pred, target_history):
        # pred: [B,1] 原TimeXer预测；target_history: [B,L] 目标闸前水位历史
        if pred.dim() != 2 or pred.shape[1] != 1:
            _shape_error("LastAnchorOutput.pred", "[B,1]", pred)
        if target_history.dim() != 2 or target_history.shape[0] != pred.shape[0]:
            _shape_error("LastAnchorOutput.target_history", f"[{pred.shape[0]},L]", target_history)
        window_mean = target_history.mean(dim=1, keepdim=True)
        last_value = target_history[:, -1:]
        # 运算顺序与06轮LastAnchor相同：原预测 + (最后实测值 − 窗口均值)
        return pred + (last_value - window_mean)


class ControlDetailTokens(nn.Module):
    """目标站控制动态细节标记（无参数）。

    选取目标站"瞬时流量"和全部"孔*实际开度"（任务A为3项，4孔站点为5项），
    取最近3步减去这3步的均值，较早的9步填0，得到 [B,L,K]。
    mode:
      raw  —— 保持训练段固定标准化尺度，不再按窗口缩放，保留变化幅度（主方案）；
      norm —— 再除以该通道12步窗口标准差，与TimeXer窗口归一化同尺度（对照）；
      zero —— 全部置0，只保留"多了K个标记位置"（对照）。
    通道按字段名查找，不写死索引，便于直接迁移到其他站点。
    """

    MODES = ("raw", "norm", "zero")

    def __init__(self, metadata, mode, patch_len=DETAIL_PATCH_LEN):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"[ControlDetailTokens] mode 只能是 {self.MODES}，收到 {mode!r}")
        features = list(metadata["features"])
        prefix = metadata["target_station"] + ":"
        flow = [i for i, n in enumerate(features) if n == prefix + "瞬时流量"]
        gates = [i for i, n in enumerate(features) if n.startswith(prefix + "孔") and n.endswith("实际开度")]
        if len(flow) != 1:
            raise ValueError(f"[ControlDetailTokens] 目标站 {prefix}瞬时流量 应恰有1列，实际找到 {len(flow)} 列")
        if not gates:
            raise ValueError(f"[ControlDetailTokens] 没有找到目标站 {prefix}孔*实际开度 字段")
        overlap = set(flow + gates) & set(metadata["time_feature_indices"])
        if overlap or metadata["target_channel"] in flow + gates:
            raise ValueError("[ControlDetailTokens] 控制通道与日历或目标水位通道重叠，字段表有误")
        # 用普通列表保存索引，不注册buffer，保证state_dict与旧权重一致
        self.indices = flow + gates
        self.names = [features[i] for i in self.indices]
        self.total_channels = int(metadata["total_channels"])
        self.mode = mode
        self.patch_len = int(patch_len)

    @property
    def count(self):
        return len(self.indices)

    def forward(self, x):
        # x: [B,L,C] 训练段标准化后的完整输入（含日历）
        if x.dim() != 3 or x.shape[-1] != self.total_channels:
            _shape_error("ControlDetailTokens.x", f"[B,L,{self.total_channels}]", x)
        if x.shape[1] <= self.patch_len:
            raise ValueError(f"[ControlDetailTokens] 窗口长度 {x.shape[1]} 必须大于分块长度 {self.patch_len}")
        controls = x[:, :, self.indices]                       # [B,L,K]
        recent = controls[:, -self.patch_len:]                 # [B,3,K]
        detail = recent - recent.mean(dim=1, keepdim=True)     # 块内和为0，对整体平移不变
        if self.mode == "norm":
            detail = detail / torch.sqrt(controls.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        elif self.mode == "zero":
            detail = torch.zeros_like(detail)
        earlier = torch.zeros_like(controls[:, :-self.patch_len])
        out = torch.cat([earlier, detail], dim=1)
        if out.shape != controls.shape:
            _shape_error("ControlDetailTokens.out", tuple(controls.shape), out)
        return out


def external_path_groups(model, weight_decay):
    """把外部嵌入与交叉注意力的参数分入 weight_decay=0 组，其余参数保持原衰减。

    返回 (Adam参数组列表, 分组记录)。记录写入运行目录，便于核查哪些参数免衰减。
    """
    exempt, remaining, exempt_names, remaining_names = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(key in "." + name for key in EXTERNAL_PATH_KEYS):
            exempt.append(p); exempt_names.append(name)
        else:
            remaining.append(p); remaining_names.append(name)
    if not exempt:
        raise ValueError("[external_path_groups] 没有匹配到外部通路参数，请检查TimeXer参数名")
    if not remaining:
        raise ValueError("[external_path_groups] 所有参数都被免衰减，分组规则有误")
    if not any(".ex_embedding." in "." + n for n in exempt_names) or not any(".cross_attention." in "." + n for n in exempt_names):
        raise ValueError("[external_path_groups] 免衰减组必须同时包含外部嵌入和交叉注意力参数")
    groups = [{"params": exempt, "weight_decay": 0.0}, {"params": remaining, "weight_decay": float(weight_decay)}]
    record = {"exempt_decay": 0.0, "remaining_decay": float(weight_decay),
              "exempt_names": exempt_names, "remaining_names": remaining_names,
              "exempt_parameters": int(sum(p.numel() for p in exempt)),
              "remaining_parameters": int(sum(p.numel() for p in remaining))}
    return groups, record


class TimeXerVariant(NativeForecast):
    """TimeXer改进变体：完整保留原TimeXer主干，按开关组合三个组件。

    输入 [B,12,C]（任务A C=17），输出 [B,1]，与v1全部模型接口相同。
    """

    def __init__(self, name, metadata, cfg):
        if name not in VARIANTS:
            raise ValueError(f"[TimeXerVariant] 未定义的变体 {name!r}，可选：{list(VARIANTS)}")
        # 先检查字段契约，避免“张量尺寸正确、目标字段却接错”的静默问题。
        features = list(metadata["features"])
        channels = int(metadata["total_channels"])
        calendar = list(metadata["time_feature_indices"])
        if len(features) != channels or len(set(features)) != channels:
            raise ValueError("[TimeXerVariant] 字段数量与通道数不一致，或字段名重复")
        if len(calendar) != 3 or len(set(calendar)) != 3 or any(i < 0 or i >= channels for i in calendar):
            raise ValueError("[TimeXerVariant] 日历必须为三个互不重复的有效通道")
        if metadata["target_channel"] in calendar or not 0 <= int(metadata["target_channel"]) < channels:
            raise ValueError("[TimeXerVariant] 目标水位通道无效或与日历重叠")
        if int(cfg["seq_len"]) != 12:
            raise ValueError("[TimeXerVariant] 当前实验固定使用12步历史，不能静默更改窗口")
        # 先按原方式构造主干，随机数消耗与v1 TimeXer相同，同一种子得到相同初始化
        super().__init__("TimeXer", metadata["total_channels"], cfg, metadata["time_feature_indices"])
        switches = VARIANTS[name]
        self.variant_name = name
        self.path_decay = switches["path_decay"]
        self.seq_len = int(cfg["seq_len"])
        self.total_channels = int(metadata["total_channels"])
        self.target_index = int(metadata["target_channel"])
        target_name = metadata["features"][self.target_index]
        if target_name != metadata["target_station"] + ":闸前水位" or self.physical_indices[-1] != self.target_index:
            raise ValueError(f"[TimeXerVariant] 目标水位必须是物理通道最后一列，当前为 {target_name}")
        self.anchor = LastAnchorOutput() if switches["anchor"] else None
        self.control = ControlDetailTokens(metadata, switches["control"]) if switches["control"] else None

    def optimizer_groups(self, weight_decay):
        """需要外部通路免衰减时返回 (参数组, 记录)；否则返回 None，按原方式统一衰减。"""
        return external_path_groups(self, weight_decay) if self.path_decay else None

    def forward(self, x):
        self.auxiliary_loss = None
        if x.dim() != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.total_channels:
            _shape_error("TimeXerVariant.x", f"[B,{self.seq_len},{self.total_channels}]", x)
        batch = x.shape[0]
        physical = x[:, :, self.physical_indices]      # [B,L,物理通道数]，目标水位在最后一列
        calendar = x[:, :, self.calendar_indices]      # [B,L,3]
        marks = calendar if self.control is None else torch.cat([calendar, self.control(x)], dim=-1)
        expected_marks = len(self.calendar_indices) + (self.control.count if self.control else 0)
        if marks.shape != (batch, self.seq_len, expected_marks):
            _shape_error("TimeXerVariant.x_mark", f"[{batch},{self.seq_len},{expected_marks}]", marks)
        out = self.network(physical, marks, None, None)  # 原TimeXer主干，[B,1,1]
        if out.shape != (batch, 1, 1):
            _shape_error("TimeXerVariant.network_out", f"[{batch},1,1]", out)
        pred = out[:, -1, -1:]                           # [B,1]
        if self.anchor is not None:
            pred = self.anchor(pred, x[:, :, self.target_index])
        if pred.shape != (batch, 1):
            _shape_error("TimeXerVariant.out", f"[{batch},1]", pred)
        return pred


def build_variant(name, metadata, cfg, device):
    """与adapters.build相同的调用方式，返回放到指定设备上的模型。"""
    return TimeXerVariant(name, metadata, cfg).to(device)
