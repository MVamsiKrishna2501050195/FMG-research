"""Compact BRFSS weighted-vs-unweighted training sensitivity check."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PUB = Path(r"C:\Users\mvk80\Documents\Codex\2026-11-05\search-lacuna\work\FMG-research\experiments\public_health")
sys.path.insert(0, str(PUB))
from run_public_health_fmg_audit import load_brfss, make_split, make_preprocessor, mask_matrix, build_model, _score, fmg_from_predictions

SEEDS = [42, 123, 456]

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--brfss", required=True); ap.add_argument("--output-dir", default="brfss_weight_sensitivity_outputs")
    args = ap.parse_args(); out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    data = load_brfss(Path(args.brfss).resolve()); rows=[]
    for seed in SEEDS:
        train_idx, test_idx = make_split(data, seed, .20, False)
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        prep = make_preprocessor(data, "BRFSS2024")
        xtr = np.asarray(prep.fit_transform(train), dtype=np.float32)
        xte = np.asarray(prep.transform(test), dtype=np.float32)
        from sklearn.impute import SimpleImputer
        imp=SimpleImputer(strategy="median"); xtr=imp.fit_transform(xtr).astype(np.float32); xte=imp.transform(xte).astype(np.float32)
        ytr=train.target.to_numpy(int); yte=test.target.to_numpy(int); g=test.group.to_numpy(int); wtr=train.weight.to_numpy(float); wte=test.weight.to_numpy(float)
        xm,_=mask_matrix(xte,xtr,g,"MCAR",.20,seed)
        for training in ("unweighted","survey_weighted"):
            m=build_model("RF",seed)
            if training=="survey_weighted": m.fit(xtr,ytr,sample_weight=wtr)
            else: m.fit(xtr,ytr)
            pc,_,_= _score(m,xte,yte,g,wte); pm,_,_= _score(m,xm,yte,g,wte)
            fmg,rl,al=fmg_from_predictions(yte,pc,pm,g,wte)
            rows.append({"dataset":"BRFSS2024","model":"RF","seed":seed,"training_weighting":training,"mask_rate":.20,"fmg_g0":fmg,"recall_loss_g0":rl,"accuracy_loss":al})
        pd.DataFrame(rows).to_csv(out/"brfss_weight_training_sensitivity_per_seed.csv",index=False,float_format="%.8f")
        print("completed seed",seed,flush=True)
    raw=pd.DataFrame(rows); raw.to_csv(out/"brfss_weight_training_sensitivity_per_seed.csv",index=False,float_format="%.8f")
    summary=raw.groupby(["dataset","model","training_weighting"],as_index=False).agg(n_seeds=("fmg_g0","size"),mean_fmg_g0=("fmg_g0","mean"),sd_fmg_g0=("fmg_g0","std"),mean_recall_loss_g0=("recall_loss_g0","mean"),mean_accuracy_loss=("accuracy_loss","mean"))
    summary.to_csv(out/"brfss_weight_training_sensitivity_summary.csv",index=False,float_format="%.8f")
    (out/"brfss_weight_training_sensitivity_manifest.txt").write_text("BRFSS 80/20 split; seeds 42,123,456; RF; 20% MCAR; CDC survey weights used for held-out evaluation in both arms; training compared unweighted vs survey-weighted. Compact sensitivity check.\n",encoding="utf-8")
    print(summary.to_string(index=False),flush=True)
if __name__=="__main__": main()
