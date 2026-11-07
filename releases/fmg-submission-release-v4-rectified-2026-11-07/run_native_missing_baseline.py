"""Resumable native missing-value baseline for FMG.

XGBoost and LightGBM are trained on clean encoded/scaled data and evaluated
on identically masked test data containing NaNs. Their native missing-value
routing is used; no test-time imputation is applied.
"""
from __future__ import annotations
import json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

ROOT=Path('/content'); REPO=ROOT/'work'/'FMG-research'
sys.path.insert(0,str(REPO/'analysis'))
from _primary_core import SEEDS, load_adult, load_compas, load_taiwan, preprocess, build_model, mcar_mask

RATE=0.20; MODELS=('XGB','LGB')
DATASETS=(('Adult Income',load_adult,ROOT/'data'/'adult.zip'),('COMPAS',load_compas,ROOT/'data'/'compas-scores-two-years.csv'),('Taiwan Credit',load_taiwan,ROOT/'data'/'taiwan_credit.csv'))
OUT=Path(os.environ.get('FMG_NATIVE_OUT',str(ROOT/'outputs'/'native_missing_baseline'))); OUT.mkdir(parents=True,exist_ok=True)
CSV=OUT/'native_missing_per_seed.csv'; JSON=OUT/'native_missing_manifest.json'

def metrics(model,x,y,g):
    pred=model.predict(x); f=(g==0)
    return {'accuracy':float(accuracy_score(y,pred)),'recall_g0':float(recall_score(y[f],pred[f],zero_division=0))}

def main():
    rows=[]
    if CSV.exists(): rows=pd.read_csv(CSV).to_dict('records'); print('resuming',len(rows),'rows',flush=True)
    done={(r['dataset'],r['model'],int(r['seed'])) for r in rows}
    for dsname,loader,path in DATASETS:
        data=loader(path)
        for seed in SEEDS:
            xtr,xte,ytr,yte,g=preprocess(data, int(seed))
            masked=mcar_mask(xte,RATE,int(seed)+10000)
            for name in MODELS:
                if (dsname,name,int(seed)) in done: continue
                t0=time.perf_counter(); model=build_model(name,int(seed)); model.fit(xtr,ytr); fit_s=time.perf_counter()-t0
                clean=metrics(model,xte,yte,g); stress=metrics(model,masked,yte,g)
                ad=clean['accuracy']-stress['accuracy']; rd=clean['recall_g0']-stress['recall_g0']
                rows.append({'dataset':dsname,'model':name,'seed':int(seed),'mask_rate':RATE,'accuracy_clean':clean['accuracy'],'accuracy_masked':stress['accuracy'],'recall_g0_clean':clean['recall_g0'],'recall_g0_masked':stress['recall_g0'],'accuracy_loss':ad,'recall_g0_loss':rd,'fmg':rd-ad,'fit_seconds':fit_s,'train_rows':len(xtr),'test_rows':len(xte),'features':xte.shape[1]})
                pd.DataFrame(rows).to_csv(CSV,index=False)
                print(dsname,seed,name,'FMG',round(rows[-1]['fmg'],5),flush=True)
    manifest={'protocol':{'models':list(MODELS),'seeds':list(SEEDS),'mask_rate':RATE,'test_only_masking':True,'test_missing_values_are_nan':True,'no_test_imputation':True,'rows_expected':3*2*10},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},'output':str(CSV)}
    JSON.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    out=pd.DataFrame(rows); print(out.groupby(['dataset','model'],as_index=False).agg(mean_fmg=('fmg','mean'),sd_fmg=('fmg','std'),positive_splits=('fmg',lambda x:int((x>0).sum())),mean_fit_seconds=('fit_seconds','mean'),n=('fmg','size')).to_string(index=False))
if __name__=='__main__': main()
