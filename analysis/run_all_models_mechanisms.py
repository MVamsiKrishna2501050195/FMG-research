"""Resumable all-model rate/mechanism sensitivity extension.

Three datasets x four classifier families x ten seeds x three rates x three
constructed mechanisms. Results are descriptive and checkpointed per dataset.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, r"C:\Users\mvk80\Documents\Codex\2026-11-05\search-lacuna\work\FMG-research\sensitivity")
from run_typeaware_imputation import load_datasets, fitted_transformer

SEEDS = [42, 123, 456, 1, 7, 13, 21, 37, 55, 99]
MODELS = ("RF", "XGBoost", "LightGBM", "MLP")


def signal(X):
    cols = []
    for c in X.columns:
        a = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
        med = np.nanmedian(a) if np.isfinite(a).any() else 0.0
        a = np.where(np.isfinite(a), a, med)
        lo, hi = np.min(a), np.max(a)
        if hi > lo: cols.append((a-lo)/(hi-lo))
    return np.mean(np.vstack(cols), axis=0) if cols else np.zeros(len(X))


def mask_frame(X, rate, seed, mechanism):
    rng = np.random.RandomState(seed)
    n, p = X.shape
    if mechanism == "MCAR":
        probs = np.full((n, p), rate)
    else:
        s = signal(X)
        mult = (0.50 + s) if mechanism == "MAR" else (0.35 + 1.30*s)
        row_p = np.clip(rate * mult, .02, .95)
        row_p = np.clip(row_p * rate / row_p.mean(), .001, .95)
        probs = np.repeat(row_p[:, None], p, axis=1)
    m = rng.rand(n, p) < probs
    out = X.copy()
    for j, c in enumerate(X.columns): out.loc[m[:, j], c] = np.nan
    return out, int(m.sum())


def score(model, X, y, g):
    pred = model.predict(X)
    g0, g1 = g == 0, g == 1
    return (float(accuracy_score(y, pred)), float(recall_score(y[g0], pred[g0], zero_division=0)),
            float(recall_score(y[g1], pred[g1], zero_division=0)))


def build_model(name, seed):
    if name == "RF": return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "XGBoost": return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=seed, eval_metric="logloss", n_jobs=4, tree_method="hist")
    if name == "LightGBM": return LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=.05, random_state=seed, verbosity=-1)
    return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=seed, early_stopping=True)


def main():
    out = Path("all_models_mechanism_outputs").resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dname, (df, features, sensitive, target, declared_cat) in load_datasets().items():
        print("starting", dname, flush=True)
        X, y, g = df[features].copy(), df[target].to_numpy(int), df[sensitive].to_numpy(int)
        cats = list(X.select_dtypes(include=["object", "category"]).columns) if declared_cat is None else list(declared_cat)
        for seed in SEEDS:
            Xt, Xv, yt, yv, gt, gv = train_test_split(X, y, g, test_size=.30, random_state=seed, stratify=y)
            masks = {(rate, mech): mask_frame(Xv, rate, seed, mech)[0]
                     for rate in (.05, .10, .20) for mech in ("MCAR", "MAR", "MNAR")}
            tr = fitted_transformer(Xt, cats, "median_mode")
            A, B = tr.fit_transform(Xt), tr.transform(Xv)
            for name in MODELS:
                m = build_model(name, seed)
                if name == "MLP":
                    sc = StandardScaler(); A_fit = sc.fit_transform(A); B_clean = sc.transform(B)
                    B_masks = {k: sc.transform(tr.transform(v)) for k,v in masks.items()}
                else:
                    A_fit, B_clean = A, B
                    B_masks = {k: tr.transform(v) for k,v in masks.items()}
                m.fit(A_fit, yt)
                clean = score(m, B_clean, yv, gv)
                for (rate, mech), Bm in B_masks.items():
                    masked = score(m, Bm, yv, gv); ad = clean[0] - masked[0]
                    rows.append({"dataset": dname, "model": name, "seed": seed, "masking_rate": rate,
                        "mechanism": mech, "accuracy_drop": ad,
                        "recall_drop_g0": clean[1]-masked[1], "fmg_g0": clean[1]-masked[1]-ad,
                        "recall_drop_g1": clean[2]-masked[2], "fmg_g1": clean[2]-masked[2]-ad})
            pd.DataFrame(rows).to_csv(out / "all_models_mechanisms_per_seed.csv", index=False, float_format="%.8f")
        print("completed", dname, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "all_models_mechanisms_per_seed.csv", index=False, float_format="%.8f")
    summary = raw.groupby(["dataset","model","mechanism","masking_rate"], as_index=False).agg(
        n_seeds=("fmg_g0","size"), mean_fmg_g0=("fmg_g0","mean"), sd_fmg_g0=("fmg_g0","std"),
        positive_fmg_g0_seeds=("fmg_g0", lambda x:int((x>0).sum())), mean_accuracy_drop=("accuracy_drop","mean"),
        mean_recall_drop_g0=("recall_drop_g0","mean"), mean_fmg_g1=("fmg_g1","mean"),
        positive_fmg_g1_seeds=("fmg_g1", lambda x:int((x>0).sum())))
    summary.to_csv(out / "all_models_mechanisms_summary.csv", index=False, float_format="%.8f")
    (out / "all_models_mechanisms_manifest.txt").write_text(
        "10 seeds; RF/XGBoost/LightGBM/MLP; 5%, 10%, 20%; MCAR and constructed MAR/MNAR.\n"
        "Median numeric + mode categorical imputation; MLP additionally standardized after transformation.\n"
        "MAR/MNAR are constructed stress scenarios and are not natural missingness validation.\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
