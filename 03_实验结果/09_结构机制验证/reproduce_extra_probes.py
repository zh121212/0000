"""重建探针证据；固定训练段拟合、验证段评分，非正式预测模型比较。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import local_weight_decay_diagnosis as prior
import torch


def residual_probes(features,packs,filename):
    rows=[]
    thresholds=np.quantile(abs(packs['train']['y']-packs['train']['persistence']),[.5,.9])
    for seed in [42,123,2024,2025,3407]:
        train=pd.read_csv(prior.OUT/'predictions'/f'TimeXer_LastAnchor_wd0_s{seed}'/'train.csv').predicted.to_numpy()
        val=pd.read_csv(prior.OUT/'predictions'/f'TimeXer_LastAnchor_wd0_s{seed}'/'val.csv').predicted.to_numpy()
        residual=packs['train']['y']-train;bias=residual.mean()
        delta=abs(packs['val']['y']-packs['val']['persistence'])
        masks={'all':np.ones(len(delta),bool),'stable':delta<=thresholds[0],
               'moderate':(delta>thresholds[0])&(delta<=thresholds[1]),'fast':delta>thresholds[1]}
        for name,x in features['train'].items():
            mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-6,sd,1.)
            z=(x-mu)/sd
            weights=np.linalg.solve(z.T@z+len(z)*.001*np.eye(z.shape[1]),z.T@(residual-bias))
            prediction=val+bias+((features['val'][name]-mu)/sd)@weights
            for group,mask in masks.items():
                rows.append({'seed':seed,'probe':name,'group':group,
                    'RMSE':float(np.sqrt(np.mean((prediction[mask]-packs['val']['y'][mask])**2))),
                    'MAE':float(np.mean(abs(prediction[mask]-packs['val']['y'][mask])))})
    pd.DataFrame(rows).to_csv(OUT/filename,index=False,encoding='utf-8-sig')


def main():
    packs={s:prior.base.load('A',s) for s in ['train','val']}
    ids=[i for i in range(17) if i not in [8,9,10]]
    normalized={};raw={};sources={}
    for split,p in packs.items():
        x=p['x'][:,:,ids].astype(float)
        z=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-5)
        coarse=np.repeat(z.reshape(len(z),4,3,14).mean(2),3,axis=1);fine=z-coarse
        normalized[split]={'coarse_patch':coarse.reshape(len(z),-1),'fine_patch':fine.reshape(len(z),-1),
            'last_patch_fine':fine[:,-3:].reshape(len(z),-1),'target_history':z[:,:,-1],
            'target_patch_fine':fine[:,:,-1],'other_fine':fine[:,:,:-1].reshape(len(z),-1),'full':z.reshape(len(z),-1)}
        coarse_raw=np.repeat(x.reshape(len(x),4,3,14).mean(2),3,axis=1);fine_raw=x-coarse_raw
        raw[split]={'raw_fine_external':fine_raw[:,:,:-1].reshape(len(x),-1),'raw_fine_all':fine_raw.reshape(len(x),-1),
            'raw_coarse_external':coarse_raw[:,:,:-1].reshape(len(x),-1),'raw_last_patch_fine_external':fine_raw[:,-3:,:-1].reshape(len(x),-1)}
        recent=fine_raw[:,-3:]
        sources[split]={'upstream':recent[:,:,:8].reshape(len(x),-1),'local_auxiliary':recent[:,:,8:13].reshape(len(x),-1),
            'target_only':recent[:,:,-1],'local_water_back_and_difference':recent[:,:,8:10].reshape(len(x),-1),
            'local_flow_and_gates':recent[:,:,10:13].reshape(len(x),-1)}
    residual_probes(normalized,packs,'patch_subspace_probes.csv')
    residual_probes(raw,packs,'raw_scale_patch_probes.csv')
    residual_probes(sources,packs,'recent_detail_source_probes.csv')
    # 原模型静态参照探针；只改变冻结评估输入，不训练、替换原模型。
    cfg,audit=prior.context();device=prior.base.setup(cfg,42)
    mean,std=prior.base.scale('A');rows=[]
    for seed in cfg['seeds']:
        prior.base.setup(cfg,seed)
        model=prior.previous.experiment_build(prior.previous.VARIANT,audit['tasks']['A'],cfg,device)
        job={'task':'A','model':prior.previous.VARIANT,'seed':seed,'lr':.001}
        model.load_state_dict(torch.load(prior.run_path(job,0)/'best.pt',map_location=device,weights_only=True))
        for split,pack in packs.items():
            x=torch.tensor(pack['x'],device=device)
            original=prior.base.predict(model,x,cfg['batch_size'])*std+mean
            flat=x.clone();flat[:,:,ids]=flat[:,-1:,ids].expand(-1,12,-1)
            reference=prior.base.predict(model,flat,cfg['batch_size'])*std+mean
            variance=x[:,:,-1].var(1,unbiased=False).cpu().numpy()
            variants={'original':original,'static_reference_difference':original-(reference-pack['persistence']),
                'output_epsilon_floor_removed':pack['persistence']+(original-pack['persistence'])*np.sqrt(variance/(variance+1e-5))}
            for name,pred in variants.items():rows+=prior.previous.score_rows(pred,pack,name,seed,split,thresholds=np.quantile(abs(packs['train']['y']-packs['train']['persistence']),[.5,.9]))
            rows.append({'model':'static_artifact_energy','seed':seed,'split':split,'group':'all',
                'RMS_static_correction_m':float(np.sqrt(np.mean((reference-pack['persistence'])**2)))})
        del model
    pd.DataFrame(rows).to_csv(OUT/'frozen_static_reference_probes.csv',index=False,encoding='utf-8-sig')
    print('Rebuilt frozen probe tables; no new formal training, no test data evaluated.')


if __name__=='__main__':main()
