"""将已核验实验转成机制、模块必要性和新颖性分开表述的中文报告。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DIAG=ROOT/'03_实验结果/09_结构机制验证'


def main():
    decision=json.loads((OUT/'decision.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed'
    data=pd.read_csv(OUT/'全部对照逐种子.csv')
    mean=data.groupby(['model','split','group'])[['MAE','MAPE','RMSE','skill_vs_persistence']].mean()
    sd=data.groupby(['model','split','group'])[['MAE','RMSE']].std()
    base='TimeXer_LastAnchor';raw=base+'_RecentRaw';norm=base+'_RecentNorm';zero=base+'_RecentZero'
    def v(name,group='all',metric='RMSE'):return float(mean.loc[(name,'val',group),metric])
    def mm(name,group='all',metric='RMSE'):return v(name,group,metric)*1000
    def reduction(old,new,group='all',metric='RMSE'):return (1-v(new,group,metric)/v(old,group,metric))*100
    timing=pd.read_csv(OUT/'timing.csv').groupby('model').seconds.agg(['median','min','max'])
    speed_ratio=float(timing.loc[raw,'median']/timing.loc[base,'median'])
    frozen=pd.read_csv(DIAG/'residual_decomposition.csv')
    stable=frozen[(frozen.condition=='LastAnchor_wd0')&(frozen.split=='val')&(frozen.group=='stable')]
    bias_fraction=float(stable.bias_squared.mean()/stable.MSE.mean())
    static=pd.read_csv(DIAG/'frozen_static_reference_probes.csv')
    static_rms=float(static[(static.model=='static_artifact_energy')&(static.split=='val')].RMS_static_correction_m.mean())
    probe=pd.read_csv(DIAG/'recent_detail_source_probes.csv').groupby(['probe','group'])[['RMSE','MAE']].mean()
    sources=pd.read_csv(OUT/'detail_source_ablations.csv').groupby('ablation')[['RMSE','delta_RMSE','prediction_change_RMS']].mean()
    accepted=decision['retain_raw_as_balanced_module']
    verdict='Accept with Revisions（工程候选可保留，研究新颖性仍需补强）' if accepted else 'Reject this version（本结构原型未通过预定目标，保留研究问题）'
    conclusion='当前数据支持保留幅值保真的最近块细节标记，作为针对剩余动态读出问题的工程候选。' if accepted else '本轮不能确认需要添加该结构：该原型未同时满足总体、平稳段、快速变化段和对照要求。'
    criteria_lines='\n'.join(f'- {key}：{"通过" if passed else "未通过"}。' for key,passed in decision['predeclared_criteria'].items())
    results=[]
    for name in ['TimeXer',base,raw,norm,zero,'Persistence']:
        sample=int(len(data[(data.model==name)&(data.split=='val')&(data.group=='all')]))
        results.append(f'- {name}：RMSE {mm(name):.6f} mm；MAE {mm(name,metric="MAE"):.6f} mm；种子/基准记录数{sample}。')
    source_lines='\n'.join(f'- {name}：去掉相应补充细节后，平均验证RMSE变化{row.delta_RMSE*1000:+.6f} mm。' for name,row in sources.iterrows())
    effective_score=6 if accepted else (4 if v(raw)<v(base) else 2)
    scores={'Higher':effective_score,'Faster':2,'Stronger':3,'Cheaper':2,'Broader':2}
    evaluation={'verdict':verdict,'paper_position':'Technique candidate; incremental, not a proven new forecasting paradigm',
        'scores':scores,'novelty_global_first_verified':False,'high_novelty_established':False,'balanced_module_retained':accepted,
        'remaining_requirements':['closest-prior method comparison','independent future-time/station confirmation','demonstrated source-specific benefit'],
        'known_resources':'RTX5060 Laptop GPU 8GB; real local taskA data; all21 new jobs completed',
        'unknown_resources':['target venue','deadline','weekly research hours','advisor cadence'],
        'fatal_flaw_risks':['F1: overlap with normalization/decomposition/multi-view prior work','F6: cross-site or strong novelty claims exceed current evidence']}
    (OUT/'novelty_evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f"""# 本地结构实验：原因、模块取舍与创新性

日期：2026-10-29。Verification Status：VERIFIED（训练、权重、数据对齐及指标已核验；不表示已通过外部泛化验证）。
本轮新增21次正式结构训练，三种条件各三学习率搜索加四个额外种子；新结果共15个正式权重、30份训练/验证预测。原TimeXer、LastAnchor及旧结果不改，未评估测试集。

## 先给结论

**{conclusion}**
原始幅值候选验证RMSE为{mm(raw):.6f} mm，相对LastAnchor的{mm(base):.6f} mm变化为下降{reduction(base,raw):.3f}%；但是否解决本轮问题必须同时看后面的预定保留条件，不能只报告总体分数。
**目前不把任何原型宣称为“已证明足够新颖”“全球首次”或“必需模块”。** 有效性、必要性、研究新颖性是不同判断。本轮对应的研究评估为：{verdict}。

## 本轮进一步排查到了什么

1. 平稳段误差主要不是固定偏置。已有LastAnchor在验证平稳段的均方误差中，各种子固定偏置平方的平均贡献约{bias_fraction*100:.3f}%。因此单纯加一个常数校准模块并不对应主要误差。
2. 用训练段拟合一个统一缩放系数，LastAnchor的最优收缩系数接近1；对所有状态一律压小修正并无充分训练证据。
3. 静态历史参照产生的多余修正RMS约{static_rms*1000:.6f} mm，相对当前平稳段误差很小。冻结模型扣除该参照、或只去掉反归一化epsilon下限，都没有带来有意义的总体改善。双前向静态参照并非当前优先方向。
4. 训练段拟合、验证段评分的残差信息探针中，绝对状态均值/尺度未带来改善，而短时动态有额外解释力。最近块来源探针中，本站流量与开度的验证RMSE为{probe.loc[('local_flow_and_gates','all'),'RMSE']*1000:.6f} mm；上游动态为{probe.loc[('upstream','all'),'RMSE']*1000:.6f} mm。**这些是诊断探针，不是公平训练的新模型指标，不能拿来替代正式实验成绩。**
5. 原模型修正量与过去变化有较强滞后关联，但不能仅凭相关峰值断言分块不可逆地丢失了信息。原分块投影本身并不必然压掉块内全部信息；证据支持的是当前训练后的短时信息读出不足。

以上残差探针均在已有骨干训练过的训练段上拟合，不是骨干的折外预测。它们用于生成结构假设，不能单独证明泛化、因果或模块必要性。验证集已多次用于诊断，本轮结构训练属于探索性实验。

## 实际设计和验证的唯一结构候选

工作名：RecentRaw，最近块幅值保真细节标记。
对13个非目标物理变量最近3步各减去该块均值，较早9步的补充通路为零。原始12步、17维输入和TimeXer全部主干仍完整保留。
这些细节由已有历史计算，使用原训练集固定标准化坐标，不再按每个窗口的标准差放大或压小。它们被送入已有的共享倒置嵌入和交叉注意力，没有增加新的预测头、编码器或可训练参数。
该细节满足块内和为零；块内加常量不改变细节；变化幅度乘常数时细节幅度也随之变化。新增标记仅改变内部信息呈现方式，不增加原始观测或未来信息。

物理字段中既有上游，也有本站闸后水位、水位差、流量和开度。除目标水位以外的变量不能全部称为“跨站点信息”；本站闸后水位与水位差还可能包含可重建的目标历史关系，不能据此夸大跨站点融合。

## 两个必要对照

- RecentNorm：同样的最近块细节，再除以窗口标准差。用来检验“保持幅值”这个机制是否成立。
- RecentZero：相同13个附加标记全部为零。用来区分真实细节作用与标记数量、共享嵌入偏置、随机训练等影响。

三者的外生标记数均为29，参数数目均为102,017；原TimeXer/LastAnchor为16个外生标记。所有主干参数名称和形状一致。三条件都从头训练，采用同样的三学习率预算、五种子、100轮上限、patience10、batch64、dropout0.1、Adam、weight_decay0和原MSE损失。
公共protocol.json未修改。结构收益只与相同weight_decay0的对照比较，不把之前修改训练规则的收益归给本次结构。

## 完整验证集结果（五种子均值，持久性为单个确定性基准）

{chr(10).join(results)}

Raw的验证RMSE种子样本标准差为{sd.loc[(raw,'val','all'),'RMSE']*1000:.6f} mm。相同种子比较中，Raw有{decision['paired_wins_vs_baseline']}/5次总体RMSE优于LastAnchor。五种子不是五个独立站点，不据此宣称统计显著。

### 分组结果

- 平稳段RMSE：LastAnchor {mm(base,'stable'):.6f}，Raw {mm(raw,'stable'):.6f}，Norm {mm(norm,'stable'):.6f}，Zero {mm(zero,'stable'):.6f} mm。
- 中等变化段RMSE：LastAnchor {mm(base,'moderate'):.6f}，Raw {mm(raw,'moderate'):.6f}，Norm {mm(norm,'moderate'):.6f}，Zero {mm(zero,'moderate'):.6f} mm。
- 快速变化段RMSE：LastAnchor {mm(base,'fast'):.6f}，Raw {mm(raw,'fast'):.6f}，Norm {mm(norm,'fast'):.6f}，Zero {mm(zero,'fast'):.6f} mm。

Raw并非每一组都最优：Norm在平稳段优于Raw，而Raw在快速变化和总体指标上更好。因此“恢复真实幅值有助于总体/快速变化预测”有对照支持，但不能进一步宣称“窗口归一化必然造成平稳段噪声，Raw在所有状态都更好”。
Raw的中等变化段RMSE相对LastAnchor仍有小幅回退，需要一并保留。平稳段也仍未优于持久性预测的{mm('Persistence','stable'):.6f} mm；前轮原衰减条件的平稳RMSE约0.884 mm同样更低，不能把本轮称为消除了所有训练规则代价。

分组只用于评估，阈值仍来自训练段真实一步变化的50%和90%分位数，不是水文标准或传感器精度。未来真实变化不作为模型输入，也不用来在线选择模型。

### 训练前固定的保留条件

{criteria_lines}

最终是否作为本轮目标下的平衡模块保留：**{'是，当前数据支持作为候选保留' if accepted else '否，不能将其包装为已解决当前目标问题的模块'}**。

### 冻结模型的来源消融

以下只屏蔽新增细节标记，原始13个非目标物理变量及目标历史仍全部输入主干；不删除原结构、不重训或覆盖权重。
{source_lines}
正的误差变化说明该新增通路在当前学习到的函数中有作用；这种干预仍可能造成内部表示分布变化，不等同水文因果证明。
来源消融若显示本站流量/开度占主要作用，只支持下一轮收缩新增通路的研究方向；本轮没有把13个细节标记删成3个后重新训练，不能把“仅保留三字段”的冻结消融当成独立训练的新模型成绩。所有原始输入字段始终保留。

## 计算代价

所有条件参数数不增加，但附加标记增加了部分嵌入和交叉注意力计算。在同一本地GPU、同一1608样本验证段、batch64、预热后5次重复中：
- LastAnchor中位评估耗时{timing.loc[base,'median']:.6f}秒。
- Raw中位评估耗时{timing.loc[raw,'median']:.6f}秒，约为前者{speed_ratio:.3f}倍。
这不是单条实时请求延迟，也不是严格FLOPs或能耗基准。若收益很小，不能因为参数数目没变就声称计算无代价。

## 目前不优先添加什么

- 更深的注意力、额外CNN/LSTM/SSM整块：当前没有证据把主要瓶颈归为容量或长期周期不足，先前还存在明确的弱路径衰减问题。
- 额外日历编码器：日历已正常传入；现有问题是利用收益不稳定，不能靠重复编码自动解决。
- 只补绝对均值/状态的分支：当前残差探针没有显示额外收益。
- 常数偏置、统一缩放、双前向静态参照：不对应当前主要剩余误差，现有冻结诊断收益有限。
- 仅按历史水位波动设硬门控：历史波动并不可靠等价于未来稳定；不能按真实未来变化门控。
- 硬套水量守恒或水力约束模块：本工程未提供完整断面、传播、侧入流等建模条件，不应虚构物理方程。

上述是“当前证据不支持优先添加”，不是证明这些技术在所有数据和任务中都无效。

## 应该添加什么，以及创新性是否足够

{conclusion}
目前最有依据的功能方向是读出本站流量/开度等短时动态，并保留其变化强度。来源探针说明这是有希望的线索，但不等于已经训练验证了“仅用三字段”的专属模块；本轮原型仍含13个非目标变量。

创新性判定：**{verdict}**。
- Higher：{scores['Higher']}/10，依据上面的正式误差与预定条件。
- Faster：{scores['Faster']}/10，未建立加速贡献，评估时间还可能增加。
- Stronger：{scores['Stronger']}/10，目前只有同站点时间划分及状态分析，尚无独立站点/未来段确认。
- Cheaper：{scores['Cheaper']}/10，没有新增参数，但未证明整体算力或数据成本下降。
- Broader：{scores['Broader']}/10，只完成当前任务A，没有跨域验证。

这些评分是证据审查，不是论文录用概率。目标期刊、截止日期、每周研究时间和导师审查安排未提供，不据此虚构能力或时间判断。本地RTX5060 Laptop 8GB已完成全部21次训练，当前工程可执行性有实际证据。
主要审稿风险是F1（与既有分解/归一化/多视图工作重叠）与F6（把单站点探索结果扩张成跨站点或强新颖性声明）。它们需要更直接的近邻方法对照和独立数据来解决。
这属于渐进式方法候选，不是已经证明的新预测范式。用新名字组合已有组件，不足以建立新颖性。

### 已核实的近邻及边界

TimeXer本身已有分块与变量级交叉注意力，不能把原有结构当新增贡献。[TimeXer](https://arxiv.org/abs/2402.19072)

最后观测参照、持久性跳连与标量门控均有先例，LastAnchor本身不能直接作为“首次”创新。[NLinear作者代码](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)，[Persistence Initialization](https://link.springer.com/article/10.1007/s10489-023-04927-4)

补回归一化丢失信息、选择应保留统计量、融合原始与归一化视图均已有研究。本次具体限定为最近块局部扰动，而非恢复全部状态；但该区别的研究新颖性仍需更全面对照。[Non-stationary Transformers](https://arxiv.org/abs/2205.14415)，[SIN](https://proceedings.mlr.press/v235/han24e.html)，[SeesawNet预印本](https://arxiv.org/abs/2605.14551)

局部动态与双预测头也不是空白方向，不能把“局部支路+Transformer”本身当独创。[PatchMixer](https://arxiv.org/abs/2310.00655)

检索不是穷尽综述或专利查新。OpenAlex接口返回429，部分页面有访问验证；这些限制已记录在相关方法与创新性边界.md中。因此本报告不保证全球首创或高新颖性已成立。

## 文件和复现

- 原因诊断入口：项目根目录local_error_mechanism_diagnosis.py。
- 附加冻结探针：09_结构机制验证/reproduce_extra_probes.py。
- 结构训练入口：项目根目录local_patch_innovation_experiment.py。
- 结果校验与来源消融：本目录verify_and_decide.py。
- 本报告重建：本目录write_research_report.py。
- 全部种子和分组：本目录全部对照逐种子.csv、结构对照结果均值.csv。
- 三条件权重、逐轮训练记录和预测分别位于training和predictions目录，不覆盖任何旧结果。

本轮没有新测试成绩；不能把上一轮LastAnchor的测试值当作新模块的测试成绩。已有测试集曾用于早期诊断，后续即使评估也应标为探索结果。模块与超参数应先固定，再用独立时段/站点检验。

统计解释检查11/11：同时披露总体与分组方向；不将种子视作站点；单时段选择范围有限；无测试筛样本；保留持续性基准；全种子报告而非最差到最好；21任务完整无失败省略；无显著性筛选；明确探针到结构的探索顺序；区分干预依赖与水文因果；未来标签仅评分、不进入预测。
"""
    (OUT/'结构必要性与创新性结论.md').write_text(report,encoding='utf-8')
    print(verdict)
    print('REPORT',OUT/'结构必要性与创新性结论.md')


if __name__=='__main__':main()
