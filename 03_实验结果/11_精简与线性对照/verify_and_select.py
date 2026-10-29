"""核验本轮结果，按事先固定规则只选一个主方案。"""
from pathlib import Path
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import local_compact_control_experiment as exp
import numpy as np
import pandas as pd
import torch


def main():
    cfg,audit=exp.context();record=exp.base.read(OUT/'completed.json');plan=exp.base.read(OUT/'plan.json')
    assert record['signature_parts']==exp.parts()
    assert exp.parent.prior.protected_files()==plan['protected_files']
    for relative,digest in plan['previous_stage_files'].items():assert exp.base.sha256(ROOT/relative)==digest
    exp.base.setup(cfg,42,'cpu');meta=audit['tasks']['A']
    template=exp.previous.experiment_build(exp.previous.VARIANT,meta,cfg,torch.device('cpu'))
    reference=template.state_dict();original_keys=set(reference)
    results=list((OUT/'training').rglob('result.json'));assert len(results)==21
    for file in results:
        result=exp.base.read(file);mode=exp.MODES[result['model']]
        assert result['fingerprint']==exp.signature()
        assert result['parameters']==(102222 if mode=='linear' else 102017)
        assert result['checkpoint_sha256']==exp.base.sha256(file.parent/'best.pt')
        state=torch.load(file.parent/'best.pt',map_location='cpu',weights_only=True)
        assert original_keys.issubset(state.keys())
        assert set(state)-original_keys==({'linear_residual.weight','linear_residual.bias'} if mode=='linear' else set())
        assert all(state[k].shape==reference[k].shape for k in original_keys)
        epochs=pd.read_csv(file.parent/'epochs.csv')
        assert len(epochs)==result['epochs_run']<=cfg['epochs']
        assert int(epochs.loc[epochs.val_RMSE.idxmin(),'epoch'])==result['best_epoch']
        np.testing.assert_allclose(epochs.val_RMSE.min(),result['best_val_RMSE'],rtol=0,atol=1e-12)
    frame=pd.read_csv(OUT/'metrics.csv');assert len(frame)==120
    assert len(pd.read_csv(OUT/'inference_ablations.csv'))==40
    timing=pd.read_csv(OUT/'timing.csv');assert len(timing)==50
    packs={s:exp.base.load('A',s) for s in ['train','val']}
    thresholds=exp.base.read(exp.previous.OUT/'plan.json')['thresholds_m'];difference=0.;files=0
    for name in exp.MODES:
        selection=exp.base.read(OUT/f'selection_{name}.json')
        assert selection['lr']==min(selection['candidates'],key=lambda r:r['best_val_RMSE'])['lr']
        for seed in cfg['seeds']:
            for split,pack in packs.items():
                df=pd.read_csv(OUT/'predictions'/f'{name}_s{seed}'/f'{split}.csv')
                np.testing.assert_array_equal(df.target_index,pack['target_index'])
                np.testing.assert_array_equal(pd.to_datetime(df.time).to_numpy(),pack['time'])
                np.testing.assert_allclose(df.observed,pack['y'],rtol=0,atol=3e-11)
                np.testing.assert_allclose(df.persistence,pack['persistence'],rtol=0,atol=3e-11)
                assert np.isfinite(df.predicted).all()
                for group,mask in exp.previous.groups(pack,thresholds).items():
                    values=exp.base.metrics(df.predicted.to_numpy()[mask],pack['y'][mask],pack['persistence'][mask])
                    row=frame[(frame.model==name)&(frame.seed==seed)&(frame.split==split)&(frame.group==group)].iloc[0]
                    assert int(row['n'])==int(mask.sum())
                    for key in ['MAE','MAPE','RMSE','skill_vs_persistence']:
                        change=abs(values[key]-row[key]);difference=max(difference,change);assert change<1e-10
                if split=='val':
                    job={'task':'A','model':name,'seed':seed,'lr':selection['lr']}
                    np.testing.assert_allclose(exp.base.metrics(df.predicted.to_numpy(),pack['y'],pack['persistence'])['RMSE'],exp.done(job)['best_val_RMSE'],rtol=0,atol=1e-11)
                files+=1
    old=pd.read_csv(exp.parent.OUT/'全部对照逐种子.csv')
    raw13='TimeXer_LastAnchor_RecentRaw';base=exp.previous.VARIANT
    old=old[old.model.isin(['TimeXer',base,raw13,'Persistence'])]
    combined=pd.concat([old,frame],ignore_index=True);assert len(combined)==248
    combined.to_csv(OUT/'本轮对照逐种子.csv',index=False,encoding='utf-8-sig')
    measures=['MAE','MAPE','RMSE','skill_vs_persistence']
    means=combined.groupby(['model','split','group'])[measures].mean()
    means.reset_index().assign(MAE_mm=lambda d:d.MAE*1000,RMSE_mm=lambda d:d.RMSE*1000).to_csv(OUT/'对照结果均值.csv',index=False,encoding='utf-8-sig')
    combined.groupby(['model','split','group'])[measures].agg(['mean','std','count']).to_csv(OUT/'summary.csv',encoding='utf-8-sig')
    latency=timing.groupby('model').seconds.agg(['median','min','max'])
    latency.to_csv(OUT/'同期推理计时.csv',encoding='utf-8-sig')
    val=combined[(combined.split=='val')&(combined.group=='all')&(combined.model!='Persistence')].pivot(index='seed',columns='model',values='RMSE')
    compact='TimeXer_LastAnchor_Control3Raw';zero='TimeXer_LastAnchor_Control3Zero';linear='TimeXer_LastAnchor_LinearBypass'
    def value(name,group='all',metric='RMSE'):return float(means.loc[(name,'val',group),metric])
    eligibility={}
    for name in [compact,linear]:
        criteria={'overall_RMSE_better_than_LastAnchor':value(name)<value(base),
            'overall_MAE_better_than_LastAnchor':value(name,metric='MAE')<value(base,metric='MAE'),
            'stable_RMSE_better_than_LastAnchor':value(name,'stable')<value(base,'stable'),
            'fast_RMSE_not_worse_than_LastAnchor':value(name,'fast')<=value(base,'fast'),
            'at_least4_of5_overall_RMSE_wins':int((val[name]<val[base]).sum())>=4}
        if name==compact:criteria['beats_same_count_zero_control']=value(name)<value(zero)
        eligibility[name]={'criteria':criteria,'eligible':all(criteria.values()),'paired_wins_vs_LastAnchor':int((val[name]<val[base]).sum())}
    assert exp.base.read(exp.parent.OUT/'decision.json')['retain_raw_as_balanced_module']
    candidates=[base,raw13]+[name for name,item in eligibility.items() if item['eligible']]
    parameters={base:102017,raw13:102017,compact:102017,zero:102017,linear:102222}
    winner=min(candidates,key=lambda name:(value(name),parameters[name],float(latency.loc[name,'median'])))
    selected_lr=exp.base.read(OUT/f'selection_{winner}.json')['lr'] if winner in exp.MODES else (.001 if winner==base else exp.base.read(exp.parent.OUT/f'selection_{raw13}.json')['lr'])
    kind={base:'anchor',raw13:'detail13',compact:'control3',linear:'linear'}[winner]
    decision={'date':'2026-10-29','primary_rule':'minimum mean five-seed validation RMSE among eligible positive candidates and prior accepted controls',
        'eligibility':eligibility,'eligible_candidates':candidates,'winner':winner,'winner_kind':kind,'selected_lr':selected_lr,'weight_decay':0.,
        'parameters':parameters[winner],'validation_RMSE_m':value(winner),'validation_MAE_m':value(winner,metric='MAE'),
        'paired_wins_vs_Raw13':int((val[winner]<val[raw13]).sum()),'winner_median_seconds_per1608':float(latency.loc[winner,'median']),
        'raw13_median_seconds_per1608':float(latency.loc[raw13,'median']),
        'compaction_success':eligibility[compact]['eligible'] and value(compact)<=value(raw13) and float(latency.loc[compact,'median'])<float(latency.loc[raw13,'median']),
        'compact_RMSE_m':value(compact),'zero3_RMSE_m':value(zero),'linear_RMSE_m':value(linear),
        'test_evaluated':False,'architecture_fairness':'original TimeXer backbone present in every positive new condition; raw17x12 observations unchanged',
        'novelty_note':'linear residuals are established techniques; effective engineering outcome does not prove strong methodological novelty'}
    exp.base.write_json(OUT/'decision.json',decision)
    verification={'status':'passed','new_checkpoints_verified':21,'predictions_recomputed':files,'metric_rows':120,'same_original_backbone':True,
        'timing_repeats_per_model':10,'interleaved_timing':True,'max_metric_recalculation_difference':difference,
        'old_files_unchanged':True,'test_evaluated':False}
    exp.base.write_json(OUT/'verification.json',verification)
    exp.base.write_json(OUT/'progress.json',{'phase':'completed_and_verified','completed_jobs':21,'total_jobs':21})
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))
    print(means.loc[(slice(None),'val',slice(None)),['RMSE','MAE']].to_string())


if __name__=='__main__':main()
