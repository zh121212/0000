"""输出一页主方案；完整对照记录另存，不把消融开关带进最终实现。"""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    decision=read(OUT/'decision.json');export=read(OUT/'clean_export.json');folder=Path(export['directory'])
    assert read(OUT/'verification.json')['status']=='passed'
    assert read(folder/'export_verification.json')['status']=='passed'
    winner=decision['winner'];kind=decision['winner_kind']
    data=pd.read_csv(OUT/'本轮对照逐种子.csv')
    means=data.groupby(['model','split','group'])[['RMSE','MAE','MAPE','skill_vs_persistence']].mean()
    timing=pd.read_csv(OUT/'timing.csv').groupby('model').seconds.median()
    base='TimeXer_LastAnchor';raw13='TimeXer_LastAnchor_RecentRaw';compact='TimeXer_LastAnchor_Control3Raw';zero='TimeXer_LastAnchor_Control3Zero';linear='TimeXer_LastAnchor_LinearBypass'
    def val(model,group='all',metric='RMSE'):return float(means.loc[(model,'val',group),metric])
    def mm(model,group='all',metric='RMSE'):return val(model,group,metric)*1000
    choices={
        'control3':('TimeXer＋LastAnchor＋3项本站控制动态细节',
            '只补充本站瞬时流量、孔1开度、孔2开度最近3步去均值后的细节；通过原共享嵌入和交叉注意力读取，保留训练尺度中的变化幅度。',
            '不采用额外10项细节标记，不采用线性补偿头。原17维输入仍全部保留。',
            '预测水位＝最后实测水位＋TimeXer在3项控制动态细节条件下预测的变化量。'),
        'linear':('TimeXer＋LastAnchor＋单层线性动态补偿',
            '原TimeXer全部保留；仅增加一个205参数、零初始化的线性残差头，物理历史先减去最后观测，日历保持原值。',
            '不采用额外细节标记、额外注意力或非线性补偿网络。',
            '预测水位＝最后实测水位＋TimeXer变化量＋线性历史动态补偿。'),
        'detail13':('TimeXer＋LastAnchor＋13项最近块幅值保真细节',
            '保留上一轮已验证的13项非目标物理变量最近块细节通路，全部原主干结构不变。',
            '本轮精简或线性方案未取得更优合格结果，因此不替换已验证版本。',
            '预测水位＝最后实测水位＋TimeXer在最近块细节条件下预测的变化量。'),
        'anchor':('TimeXer＋LastAnchor',
            '仅以最后实测水位为输出基准，保持原TimeXer全部主干。',
            '不新增细节标记或线性头；当前实验没有支持更多结构。',
            '预测水位＝最后实测水位＋TimeXer预测变化量。')}
    title,addition,removed,equation=choices[kind]
    primary_gain=(1-val(winner)/val(base))*100
    vs13=(val(winner)/val(raw13)-1)*100
    latency_gain=(1-float(timing[winner])/float(timing[raw13]))*100
    command=f"& 'D:/anaconda3/envs/rl_case1/python.exe' -X utf8 '{folder.as_posix()}/predict.py' --seed 42"
    text=f"""# 最终主方案

**采用：{title}。**

{equation}

**结构只保留这些内容：** {addition}

**本轮不采纳：** {removed}

**固定设置：** 12步历史→30分钟后单步水位；batch64；Adam；weight_decay=0；学习率={decision['selected_lr']:g}；dropout0.1；最多100轮；patience10；种子42/123/2024/2025/3407。输入、标准化、划分与损失不变。

**结果均为五种子验证集均值，不是测试成绩：**

- RMSE：{mm(winner):.6f} mm；原LastAnchor为{mm(base):.6f} mm，降低{primary_gain:.3f}%。
- MAE：{mm(winner,metric='MAE'):.6f} mm；原LastAnchor为{mm(base,metric='MAE'):.6f} mm。
- 相对原13标记版：RMSE变化{vs13:+.3f}%；同期批量评估耗时变化{-latency_gain:+.3f}%。
- 参数：{decision['parameters']:,}；1608条验证样本、batch64的中位评估耗时{float(timing[winner]):.6f}秒，原13标记版为{float(timing[raw13]):.6f}秒。

**为什么采用它：** 同数量全零对照的RMSE为{mm(zero):.6f} mm，真实3项细节为{mm(compact):.6f} mm，支持保留这3项细节。

**为什么不采用线性头：** 它的总体RMSE更低（{mm(linear):.6f} mm），但平稳段RMSE由LastAnchor的{mm(base,'stable'):.6f} mm升至{mm(linear,'stable'):.6f} mm，增加{(val(linear,'stable')/val(base,'stable')-1)*100:.2f}%，未通过训练前规定的“平稳段不退化”门槛。本方案不是全部候选中总体RMSE最低者，而是通过门槛后的最优者。

**仍有取舍：** 本方案中等变化段RMSE为{mm(winner,'moderate'):.6f} mm，LastAnchor为{mm(base,'moderate'):.6f} mm；并非每个子组都提升。同期耗时仅点估计变化{-latency_gain:+.3f}%，不据此宣称有明确加速收益。

**选择依据：** 先检查总体、平稳段、快速变化段和种子一致性，再按固定的五种子平均验证RMSE选最优；没有事后换指标。

**创新结论：** 这是当前数据支持的明确工程方案。精简标记、LastAnchor或普通线性补偿本身都不能单独写成强新颖性，当前尚不能宣称全球首次或论文创新已充分成立。

**运行：**

```powershell
{command}
```

默认输入项目现有验证包；输出prediction_seed42.csv，水位单位m。42是固定演示种子，不是根据测试成绩挑出的最好种子。五个种子分别保存，不是集成预测。
自有输入使用`--input 路径.npz`，其中x须为[B,12,17]；物理变量未标准化时再加`--raw`。字段顺序见config.json；日历仍须按原规则编码为0至1。
日历编码：time_of_day=(小时×2+分钟/30)/47；weekday=星期索引/6（周一0）；week_of_month=floor((日-1)/7)/4。

目录只含选定结构model.py、预测入口predict.py、config.json、原训练scaler.npz、五个权重和校验凭据。原工程默认入口和Baseline结果没有被替换。
"""
    (folder/'最终方案.md').write_text(text,encoding='utf-8')
    (OUT/'最终方案_一页版.md').write_text(text,encoding='utf-8')
    rows=[]
    for model in [base,raw13,compact,zero,linear,'Persistence']:
        for group in ['all','stable','moderate','fast']:
            rows.append({'model':model,'group':group,'RMSE_mm':mm(model,group),'MAE_mm':mm(model,group,'MAE')})
    pd.DataFrame(rows).to_csv(OUT/'重点对照.csv',index=False,encoding='utf-8-sig')
    ablations=pd.read_csv(OUT/'inference_ablations.csv')
    ablation_means=ablations[ablations.group=='all'].groupby('model').RMSE.mean()
    detail_disabled=float(ablation_means[compact])*1000
    linear_disabled=float(ablation_means[linear])*1000
    training_lines=[]
    for name in [base,raw13,compact,linear]:
        train=float(means.loc[(name,'train','all'),'RMSE'])*1000
        training_lines.append(f'- {name}：训练RMSE {train:.6f} mm；验证RMSE {mm(name):.6f} mm。')
    criterion=[]
    for model,info in decision['eligibility'].items():
        criterion.append(f"- {model}：{'合格' if info['eligible'] else '不合格'}，逐项条件：{json.dumps(info['criteria'],ensure_ascii=False)}。")
    result_lines=[]
    for model in [base,raw13,compact,zero,linear,'Persistence']:
        result_lines.append(f"- {model}：验证RMSE {mm(model):.6f} mm，MAE {mm(model,metric='MAE'):.6f} mm。")
    report=f"""# 精简与线性对照：完整依据

日期2026-10-29，Verification Status：VERIFIED。新增21次训练，三条件各7次，最终15个正式种子权重。仅使用训练/验证段，未计算测试指标。

## 唯一主方案

{title}。模型标识：`{winner}`。
按事先固定的验证RMSE规则选择，不按速度、某个种子或某个子组临时换主指标。最终独立目录：{folder.as_posix()}。

## 比较对象

1. Control3Raw：仅新增本站流量和两孔开度的3个细节标记。原17维观测和TimeXer全部结构不删，参数102,017。
2. Control3Zero：同样新增3个标记，但附加值全零，排除仅由标记数量/随机训练带来的影响。参数102,017。
3. LinearBypass：原TimeXer和LastAnchor全部保留，增加一个204输入、1输出的线性补偿头，零初始化，与骨干联合训练。物理历史减去最后观测，日历不变。参数102,222。

线性对照是独立的对照条件，不是删掉TimeXer后换成线性模型。它能判断普通动态补偿是否足以解释收益。

## 公共条件

所有条件均使用原17维输入、12步历史、单步输出、原时间划分及原训练标准化；MSE损失、Adam、batch64、dropout0.1、weight_decay0、梯度裁剪5、最多100轮、patience10。
每个条件按种子42搜索0.0001/0.0003/0.001，选验证RMSE最小者，再训练其余4种子。原13标记版和无新增模块LastAnchor使用此前相同训练条件的结果，旧权重、源码和预测已核对，均未覆盖。

## 总体结果

{chr(10).join(result_lines)}

## 预定门槛与选择

{chr(10).join(criterion)}

三标记精简是否同时满足“RMSE不高于13标记版、同期计时更低、有效性条件合格”：{decision['compaction_success']}。
主方案相对LastAnchor验证RMSE降低{primary_gain:.3f}%；相对13标记版的RMSE变化{vs13:+.3f}%，同期批量耗时变化{-latency_gain:+.3f}%。
完整分组与种子见重点对照.csv、本轮对照逐种子.csv。分组阈值只来自训练标签分位数，仅评分使用，不在预测时使用未来真实变化。

## 本轮验证了什么

- 同为3个附加位置，真实细节版验证RMSE为{mm(compact):.6f} mm，全零版为{mm(zero):.6f} mm。只有这个比较支持时，才能将收益归于附加信息，而非单纯多了标记位置。
- 保持训练好的参数不动，关闭3项细节后验证RMSE为{detail_disabled:.6f} mm，完整模型为{mm(compact):.6f} mm；关闭线性补偿头后为{linear_disabled:.6f} mm，完整线性模型为{mm(linear):.6f} mm。
- 关闭分支只检验已训练模型对分支的依赖，不能代替重训对照，也不证明水文因果。
- 普通线性补偿与3项细节并列训练，避免把本可由简单模型解释的收益归为复杂模块的新颖贡献。主方案仍由预定规则决定。

训练/验证误差如下，不能只凭二者数值不同就判定过拟合，因为两个时间段的水位波动不同：

{chr(10).join(training_lines)}

## 计算计时

本轮重新在同一GPU上预热并轮换五个模型的测量顺序，每模型重复10次，数据相同1608条验证样本，batch64；不沿用上一轮不同时间的计时。它是批量评估耗时，不是单条实时延迟或FLOPs。
参数少或标记少不自动等于更快，实际测量见同期推理计时.csv。主方案相对13标记版约0.8%的中位数差距很小，本轮不宣称具有可重复的实质速度优势。

## 交付与校验

21个新权重、30份新预测、120行分组指标及50次计时记录已检查；同一个TimeXer主干所有原参数名和形状保留。线性对照只多一个205参数头。
最终model.py不导入实验编排脚本，只读取选定config并使用原vendor TimeXer；严格加载五个权重，完整验证段与原实验预测对齐，batch1与batch64一致。详见最终目录export_verification.json和provenance.json。

## 创新性和边界

这轮是在确定一个可直接使用的最简工程方案，不会把“精简标记”“LastAnchor”或“单层线性补偿”本身包装成强新颖性。先行工作与上一轮的新颖性审查结论仍适用。
五个种子描述训练随机性，不代表五个独立站点；验证段已经被用于多轮诊断，本轮结果是探索性工程证据，不能替代后续独立时段/站点确认。
已有测试集曾用于早期诊断，本轮没有再次查看测试成绩，也不能将旧模型测试值赋给本轮新结构。

统计解释检查11/11：总体与分组共同报告；种子不冒充站点；披露单时段范围；不按测试结果筛样本或控制变量；保留持续性基准；全种子而非最差到最好；21任务无失败省略；无显著性筛选；训练前固定比较规则；区分结构对照与水文因果；未来标签仅评分。
"""
    (OUT/'完整对照依据.md').write_text(report,encoding='utf-8')
    print('FINAL',winner)
    print('ONE_PAGE',folder/'最终方案.md')


if __name__=='__main__':main()
