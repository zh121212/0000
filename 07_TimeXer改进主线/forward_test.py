"""随机输入Forward Test：只用随机张量和字段表，不读数据包，不训练，不看任何成绩。

检查内容：
R1 每个变体前向/反向，输出 [B,1]，梯度有限；输入未被就地修改；
   另用"目标站4孔"的假字段表再跑一遍，确认组件不依赖任务A的通道数；
R2 数值性质：LastAnchor在头部输出为0时等于最后实测值；Zero全零；
   Raw只在最近3步非零且块内和为0；Raw与Norm只差窗口标准差；
R3 外部通路免衰减的参数分组：非空、无重复、覆盖全部参数、只含外部嵌入和交叉注意力；
R6 错误形状会给出中文报错。
运行：python -X utf8 07_TimeXer改进主线/forward_test.py
"""
from pathlib import Path
import json
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from timexer_variants import (VARIANTS, ControlDetailTokens, LastAnchorOutput,  # noqa: E402
                              TimeXerVariant, EXTERNAL_PATH_KEYS)


def task_a_metadata():
    audit = json.loads((ROOT / "01_数据集/03_划分打包/packing_audit.json").read_text(encoding="utf-8"))
    return audit["tasks"]["A"]


def fake_four_gate_metadata():
    """假想站点：上游8个物理量 + 日历3个 + 目标站8个物理量（4孔），共19通道。"""
    up = [f"up:{n}" for n in ["闸前水位", "闸后水位", "闸前闸后水位差", "瞬时流量",
                              "孔1实际开度", "孔2实际开度", "孔3实际开度", "孔4实际开度"]]
    cal = ["time_of_day", "weekday", "week_of_month"]
    local = [f"down:{n}" for n in ["闸后水位", "闸前闸后水位差", "瞬时流量",
                                   "孔1实际开度", "孔2实际开度", "孔3实际开度", "孔4实际开度", "闸前水位"]]
    features = up + cal + local
    return {"features": features, "target_station": "down", "total_channels": len(features),
            "target_channel": len(features) - 1, "time_feature_indices": [8, 9, 10]}


def config():
    cfg = json.loads((ROOT / "04_实验配置/protocol.json").read_text(encoding="utf-8"))
    return {"seq_len": cfg["seq_len"], "dropout": cfg["dropout"]}


def random_input(meta, batch=8, seq_len=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, meta["total_channels"], generator=g)
    x[:, :, meta["time_feature_indices"]] = torch.rand(batch, seq_len, 3, generator=g)  # 日历为0–1
    return x


def check_variant(name, meta, cfg, records):
    torch.manual_seed(42)
    model = TimeXerVariant(name, meta, cfg)
    x = random_input(meta)
    before = x.clone()
    model.train()
    out = model(x)
    assert out.shape == (x.shape[0], 1), f"{name} 输出形状 {tuple(out.shape)}"
    torch.testing.assert_close(x, before, rtol=0, atol=0)  # 输入未被就地修改
    out.square().mean().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values()), f"{name} 梯度无效"
    ex_grad = sum(float(g.abs().sum()) for n, g in grads.items() if ".ex_embedding." in "." + n)
    assert ex_grad > 0, f"{name} 外部嵌入没有收到梯度，外部信息未进入预测"
    marks = 3 + (model.control.count if model.control else 0)
    groups = model.optimizer_groups(1e-4)
    if VARIANTS[name]["path_decay"]:
        params, record = groups
        ids = [id(p) for g in params for p in g["params"]]
        assert len(ids) == len(set(ids)) == sum(1 for _ in model.parameters()), f"{name} 参数分组重复或遗漏"
        assert all(any(k in "." + n for k in EXTERNAL_PATH_KEYS) for n in record["exempt_names"])
        exempt = record["exempt_parameters"]
    else:
        assert groups is None
        exempt = 0
    records.append({"model": name, "channels": meta["total_channels"], "x_mark_columns": marks,
                    "parameters": sum(p.numel() for p in model.parameters()), "exempt_parameters": exempt,
                    "output_shape": list(out.shape), "finite_gradients": True})
    return model


def check_components(meta, cfg):
    x = random_input(meta, seed=1)
    # LastAnchorOutput：头部输出为0时，预测等于最后实测值
    hist = x[:, :, meta["target_channel"]]
    zero_head = hist.mean(1, keepdim=True)  # 原TimeXer在头部输出为0时给出窗口均值
    torch.testing.assert_close(LastAnchorOutput()(zero_head, hist), hist[:, -1:], rtol=0, atol=1e-6)
    # 整模型：把预测头清零，LA变体应输出最后实测值
    torch.manual_seed(0)
    model = TimeXerVariant("TimeXer_LA_PD_C3Raw", meta, cfg).eval()
    with torch.no_grad():
        model.network.head.linear.weight.zero_(); model.network.head.linear.bias.zero_()
        torch.testing.assert_close(model(x), hist[:, -1:], rtol=0, atol=1e-5)
    raw, norm, zero = (ControlDetailTokens(meta, m) for m in ("raw", "norm", "zero"))
    r, n, z = raw(x), norm(x), zero(x)
    assert r.shape == (x.shape[0], x.shape[1], raw.count)
    torch.testing.assert_close(z, torch.zeros_like(z), rtol=0, atol=0)
    torch.testing.assert_close(r[:, :-3], torch.zeros_like(r[:, :-3]), rtol=0, atol=0)
    torch.testing.assert_close(r[:, -3:].sum(1), torch.zeros_like(r[:, 0]), rtol=0, atol=1e-5)
    controls = x[:, :, raw.indices]
    std = torch.sqrt(controls.var(1, keepdim=True, unbiased=False) + 1e-5)
    torch.testing.assert_close(n, r / std, rtol=1e-5, atol=1e-6)
    # 某个开度最后一步加常数c：最近块细节按 (-c/3, -c/3, 2c/3) 变化，较早9步不变
    shifted = x.clone(); gate = raw.indices[1]; c = 0.9
    shifted[:, -1, gate] += c
    delta = raw(shifted) - r
    expect = torch.zeros_like(delta); expect[:, -3:, 1] = torch.tensor([-c / 3, -c / 3, 2 * c / 3])
    torch.testing.assert_close(delta, expect, rtol=0, atol=1e-5)
    # 最近3步整体加同一常数：细节不变（只反映块内变化，不反映绝对水平）
    lifted = x.clone(); lifted[:, -3:, raw.indices] += 1.7
    torch.testing.assert_close(raw(lifted), r, rtol=0, atol=1e-5)
    return raw.names


def check_equivalence_and_batch(meta, cfg):
    """组件全关（TimeXer_PD推理时等同原TimeXer）应与原适配器逐样本一致；批量1与批量8结果一致。"""
    from adapters import NativeForecast
    torch.manual_seed(7)
    original = NativeForecast("TimeXer", meta["total_channels"], cfg, meta["time_feature_indices"]).eval()
    variant = TimeXerVariant("TimeXer_PD", meta, cfg).eval()
    variant.load_state_dict(original.state_dict(), strict=True)  # 参数名与形状完全相同
    x = random_input(meta, seed=3)
    with torch.no_grad():
        torch.testing.assert_close(variant(x), original(x), rtol=0, atol=1e-6)
        for name in VARIANTS:
            model = TimeXerVariant(name, meta, cfg).eval()
            one = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])])
            assert one.shape == (x.shape[0], 1) and torch.isfinite(one).all(), f"{name} 批量1输出异常"
            torch.testing.assert_close(one, model(x), rtol=1e-5, atol=1e-5)


def check_errors(meta, cfg):
    model = TimeXerVariant("TimeXer_LA_PD_C3Raw", meta, cfg)
    bad = torch.randn(2, 12, meta["total_channels"] - 1)
    try:
        model(bad)
    except ValueError as error:
        assert "形状" in str(error)
    else:
        raise AssertionError("错误形状没有被拦截")
    try:
        TimeXerVariant("TimeXer_Unknown", meta, cfg)
    except ValueError as error:
        assert "未定义" in str(error)
    else:
        raise AssertionError("未知变体名没有被拦截")


def main():
    cfg = config()
    results = {}
    for label, meta in (("task_A", task_a_metadata()), ("fake_4_gates", fake_four_gate_metadata())):
        records = []
        for name in VARIANTS:
            check_variant(name, meta, cfg, records)
        control_names = check_components(meta, cfg)
        check_equivalence_and_batch(meta, cfg)
        check_errors(meta, cfg)
        results[label] = {"variants": records, "control_fields": control_names}
        print(f"FORWARD TEST PASS {label}: {len(records)} 个变体，控制细节字段 {control_names}", flush=True)
    expected = ["shierlihe:瞬时流量", "shierlihe:孔1实际开度", "shierlihe:孔2实际开度"]
    assert results["task_A"]["control_fields"] == expected, results["task_A"]["control_fields"]
    assert len(results["fake_4_gates"]["control_fields"]) == 5
    out = ROOT / "03_实验结果/12_TimeXer改进主线/forward_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "passed", **results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部通过，记录已写入 {out}")


if __name__ == "__main__":
    main()
