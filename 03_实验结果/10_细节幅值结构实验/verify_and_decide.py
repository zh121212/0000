"""结构实验的完整性、消融、预定保留条件及结果汇总。"""
from pathlib import Path
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import local_patch_innovation_experiment as exp
import numpy as np
import pandas as pd
import torch


def main():
    cfg,audit=exp.context()
    completed=exp.base.read(OUT/'completed.json')
    assert completed['signature_parts']==exp.parts()
    assert exp.prior.protected_files()==exp.base.read(OUT/'plan.json')['protected_files']
    frozen=exp.base.read(ROOT/'03_实验结果/09_结构机制验证/frozen_diagnosis.json')
    for relative,digest in frozen['source_sha256'].items():assert exp.base.sha256(ROOT/relative)==digest
    results=list((OUT/'training').rglob('result.json'));assert len(results)==21
    exp.base.setup(cfg,42,'cpu')
    template=exp.previous.experiment_build(exp.previous.VARIANT,audit['tasks']['A'],cfg,torch.device('cpu'))
    for file in results:
        result=exp.base.read(file)
        assert result['fingerprint']==exp.signature() and result['parameters']==102017
        assert exp.base.sha256(file.parent/'best.pt')==result['checkpoint_sha256']
        state=torch.load(file.parent/'best.pt',map_location='cpu',weights_only=True)
        assert state.keys()==template.state_dict().keys()
        assert all(value.shape==template.state_dict()[key].shape for key,value in state.items())
        epochs=pd.read_csv(file.parent/'epochs.csv')
        assert len(epochs)==result['epochs_run']<=cfg['epochs']
        assert int(epochs.loc[epochs.val_RMSE.idxmin(),'epoch'])==result['best_epoch']
        np.testing.assert_allclose(epochs.val_RMSE.min(),result['best_val_RMSE'],rtol=0,atol=1e-12)
    frame=pd.read_csv(OUT/'metrics.csv');assert len(frame)==120
    assert len(pd.read_csv(OUT/'inference_ablations.csv'))==40
    assert len(pd.read_csv(OUT/'timing.csv'))==25
    packs={s:exp.base.load('A',s) for s in ['train','val']}
    thresholds=exp.base.read(exp.previous.OUT/'plan.json')['thresholds_m']
    difference=0.;prediction_files=0
    for name in exp.MODES:
        selection=exp.base.read(OUT/f'selection_{name}.json')
        assert selection['lr']==min(selection['candidates'],key=lambda row:row['best_val_RMSE'])['lr']
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
                    for key in ['RMSE','MAE','MAPE','skill_vs_persistence']:
                        current=abs(row[key]-values[key]);difference=max(difference,current);assert current<1e-10
                if split=='val':
                    job={'task':'A','model':name,'seed':seed,'lr':selection['lr']}
                    np.testing.assert_allclose(exp.base.metrics(df.predicted.to_numpy(),pack['y'],pack['persistence'])['RMSE'],exp.done(job)['best_val_RMSE'],rtol=0,atol=1e-11)
                prediction_files+=1
    controls=pd.read_csv(exp.prior.OUT/'metrics.csv')
    controls=controls[(controls.weight_decay==0)|(controls.model=='Persistence')].copy()
    combined=pd.concat([controls,frame],ignore_index=True)
    assert len(combined)==208
    combined.to_csv(OUT/'全部对照逐种子.csv',index=False,encoding='utf-8-sig')
    metrics=['MAE','MAPE','RMSE','skill_vs_persistence']
    means=combined.groupby(['model','split','group'])[metrics].mean()
    stds=combined.groupby(['model','split','group'])[metrics].std()
    means.reset_index().assign(MAE_mm=lambda d:d.MAE*1000,RMSE_mm=lambda d:d.RMSE*1000).to_csv(OUT/'结构对照结果均值.csv',index=False,encoding='utf-8-sig')
    combined.groupby(['model','split','group'])[metrics].agg(['mean','std','count']).to_csv(OUT/'summary.csv',encoding='utf-8-sig')
    raw='TimeXer_LastAnchor_RecentRaw';norm='TimeXer_LastAnchor_RecentNorm';zero='TimeXer_LastAnchor_RecentZero';baseline=exp.previous.VARIANT
    def val(name,group='all',metric='RMSE'):return float(means.loc[(name,'val',group),metric])
    paired=combined[(combined.split=='val')&(combined.group=='all')&(combined.model!='Persistence')].pivot(index='seed',columns='model',values='RMSE')
    wins=int((paired[raw]<paired[baseline]).sum())
    criteria={'overall_RMSE_better_than_baseline':val(raw)<val(baseline),'overall_MAE_better_than_baseline':val(raw,metric='MAE')<val(baseline,metric='MAE'),
        'stable_RMSE_better_than_baseline':val(raw,'stable')<val(baseline,'stable'),'fast_RMSE_not_worse_than_baseline':val(raw,'fast')<=val(baseline,'fast'),
        'overall_RMSE_better_than_norm_control':val(raw)<val(norm),'overall_RMSE_better_than_zero_control':val(raw)<val(zero),
        'at_least_4_of_5_paired_RMSE_wins':wins>=4}
    decision={'date':'2026-10-29','predeclared_criteria':criteria,'retain_raw_as_balanced_module':all(criteria.values()),'paired_wins_vs_baseline':wins,
        'test_evaluated':False,'novelty_verified_as_global_first':False,'parameter_count_all_conditions':102017,
        'note':'negative-control and amplitude-ablation results must be reported; novelty and causal mechanism are separate from raw metric gains'}
    exp.base.write_json(OUT/'decision.json',decision)
    # 只在训练结束后进行冻结标记来源消融，不改变原物理历史或校验权重。
    device=exp.base.setup(cfg,42);x=torch.tensor(packs['val']['x'],device=device);mean,scale=exp.base.scale('A');source_rows=[];source_groups=[];reproduction=[]
    for seed in cfg['seeds']:
        exp.base.setup(cfg,seed)
        baseline_model=exp.previous.experiment_build(baseline,audit['tasks']['A'],cfg,device)
        job={'task':'A','model':baseline,'seed':seed,'lr':.001}
        baseline_model.load_state_dict(torch.load(exp.prior.run_path(job,0)/'best.pt',map_location=device,weights_only=True))
        base_prediction=exp.base.predict(baseline_model,x,cfg['batch_size'])*scale+mean
        stored=pd.read_csv(exp.prior.OUT/'predictions'/f'{baseline}_wd0_s{seed}'/'val.csv').predicted.to_numpy()
        discrepancy=float(np.max(abs(base_prediction-stored)));assert discrepancy<1e-6
        reproduction.append({'seed':seed,'max_prediction_difference_m':discrepancy})
        del baseline_model
        lr=exp.base.read(OUT/f'selection_{raw}.json')['lr'];job={'task':'A','model':raw,'seed':seed,'lr':lr}
        model=exp.build(raw,audit['tasks']['A'],cfg,device);model.load_state_dict(torch.load(exp.path(job)/'best.pt',map_location=device,weights_only=True))
        clean=exp.base.predict(model,x,cfg['batch_size'])*scale+mean
        original_detail=model.detail
        for name,drop in [('drop_upstream',list(range(8))),('drop_local_water',[8,9]),('drop_local_flow_gates',[10,11,12]),('keep_local_flow_gates_only',list(range(10)))]:
            def changed(inp,drop=drop):
                detail=original_detail(inp).clone();detail[:,:,drop]=0;return detail
            model.detail=changed
            prediction=exp.base.predict(model,x,cfg['batch_size'])*scale+mean
            scores=exp.base.metrics(prediction,packs['val']['y'],packs['val']['persistence'])
            source_rows.append({'seed':seed,'ablation':name,'RMSE':scores['RMSE'],'delta_RMSE':scores['RMSE']-exp.base.metrics(clean,packs['val']['y'],packs['val']['persistence'])['RMSE'],
                'prediction_change_RMS':float(np.sqrt(np.mean((prediction-clean)**2)))})
            for group,mask in exp.previous.groups(packs['val'],thresholds).items():
                changed_score=exp.base.metrics(prediction[mask],packs['val']['y'][mask],packs['val']['persistence'][mask])
                clean_score=exp.base.metrics(clean[mask],packs['val']['y'][mask],packs['val']['persistence'][mask])
                source_groups.append({'seed':seed,'ablation':name,'group':group,'RMSE':changed_score['RMSE'],
                    'MAE':changed_score['MAE'],'delta_RMSE':changed_score['RMSE']-clean_score['RMSE']})
        model.detail=original_detail;del model
    exp.previous.save_csv(OUT/'detail_source_ablations.csv',source_rows)
    exp.previous.save_csv(OUT/'detail_source_group_ablations.csv',source_groups)
    exp.previous.save_csv(OUT/'baseline_reproduction.csv',reproduction)
    verification={'status':'passed','new_checkpoints_verified':21,'prediction_files_recomputed':prediction_files,'metric_rows':120,
        'source_ablation_cases':20,'baseline_checkpoints_reproduced':5,'same_backbone_parameter_keys_shapes':True,
        'parameters':102017,'max_metric_recalculation_difference':difference,'old_protected_files_unchanged':True,'test_evaluated':False}
    exp.base.write_json(OUT/'verification.json',verification)
    exp.base.write_json(OUT/'progress.json',{'phase':'completed_and_verified','completed_jobs':21,'total_jobs':21,'test_evaluated':False})
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))
    print(means.loc[(slice(None),'val',slice(None)),['RMSE','MAE']].to_string())


if __name__=='__main__':main()
