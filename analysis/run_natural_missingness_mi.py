"""Independent natural-missingness check using UCI myocardial infarction data."""
from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEEDS=[42,123,456,1,7,13,21,37,55,99]

def metric(y,p,g):
    f=g==0; c=g==1
    return {"accuracy":accuracy_score(y,p),"focus_recall":recall_score(y[f],p[f],zero_division=0),"comparison_recall":recall_score(y[c],p[c],zero_division=0)}

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default='natural_missingness_mi_outputs'); args=ap.parse_args(); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    ds=fetch_ucirepo(id=579); X=ds.data.features.copy(); y=ds.data.targets['FIBR_PREDS'].astype(int).to_numpy(); g= X['SEX'].astype(int).to_numpy()
    X=X.drop(columns=['SEX']).apply(pd.to_numeric,errors='coerce')
    rows=[]
    for seed in SEEDS:
        tr,te=train_test_split(np.arange(len(X)),test_size=.30,random_state=seed,stratify=y)
        Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]; gte=g[te]
        missing_count=Xte.isna().sum(axis=1).to_numpy()
        # The UCI records are broadly incomplete; use prespecified lower- and
        # upper-missingness strata rather than inventing a complete-case arm.
        cutoff=float(np.median(missing_count))
        low=missing_count<=cutoff; high=missing_count>cutoff
        for name,model in [
            ('RF',RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=-1)),
            ('XGBoost',XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=seed,eval_metric='logloss',n_jobs=4,tree_method='hist')),
            ('LightGBM',LGBMClassifier(n_estimators=100,num_leaves=31,learning_rate=.05,random_state=seed,verbosity=-1)),
            ('MLP',Pipeline([('scale',StandardScaler()),('model',MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,early_stopping=True,random_state=seed))]))]:
            prep=SimpleImputer(strategy='median'); A=prep.fit_transform(Xtr); B=prep.transform(Xte); model.fit(A,ytr); p=model.predict(B)
            allm=metric(yte,p,gte)
            for label,mask in [('low_missingness',low),('high_missingness',high)]:
                if mask.sum()==0: continue
                mm=metric(yte[mask],p[mask],gte[mask]); rows.append({'dataset':'UCI_MI_Complications','target':'FIBR_PREDS','model':name,'seed':seed,'subset':label,'n_rows':int(mask.sum()),'n_missing_mean':float(missing_count[mask].mean()),'accuracy':float(mm['accuracy']),'focus_recall':float(mm['focus_recall']),'comparison_recall':float(mm['comparison_recall']),'overall_accuracy':float(allm['accuracy'])})
        pd.DataFrame(rows).to_csv(out/'natural_missingness_mi_per_seed.csv',index=False,float_format='%.8f')
        print('completed seed',seed,flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(out/'natural_missingness_mi_per_seed.csv',index=False,float_format='%.8f')
    wide=raw.pivot_table(index=['dataset','target','model','seed'],columns='subset',values=['n_rows','n_missing_mean','accuracy','focus_recall','comparison_recall']).reset_index(); wide.columns=['_'.join([str(x) for x in col if str(x)!='']).strip('_') for col in wide.columns]
    for metric_name in ('accuracy','focus_recall','comparison_recall'):
        wide[f'{metric_name}_high_minus_low']=wide.get(f'{metric_name}_high_missingness',np.nan)-wide.get(f'{metric_name}_low_missingness',np.nan)
    wide.to_csv(out/'natural_missingness_mi_paired.csv',index=False,float_format='%.8f')
    summary=wide.groupby(['dataset','target','model'],as_index=False).agg(n_seeds=('seed','size'),mean_accuracy_gap=('accuracy_high_minus_low','mean'),sd_accuracy_gap=('accuracy_high_minus_low','std'),mean_focus_recall_gap=('focus_recall_high_minus_low','mean'),sd_focus_recall_gap=('focus_recall_high_minus_low','std'),mean_comparison_recall_gap=('comparison_recall_high_minus_low','mean'),mean_high_missing_rows=('n_rows_high_missingness','mean'),mean_low_missing_rows=('n_rows_low_missingness','mean'))
    summary.to_csv(out/'natural_missingness_mi_summary.csv',index=False,float_format='%.8f')
    (out/'natural_missingness_mi_manifest.txt').write_text('UCI Myocardial Infarction Complications, 1,700 records; target FIBR_PREDS; sex is audit group (female focus=0, male comparison=1).\nObserved feature missingness is retained and measured before train-fitted median imputation. Results compare lower- versus higher-missingness test strata split at the per-seed median missing-feature count; this is not a causal intervention or synthetic-mask experiment.\n',encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__': main()
