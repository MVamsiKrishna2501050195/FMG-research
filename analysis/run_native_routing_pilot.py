"""Five-seed native-NaN routing pilot for LightGBM and XGBoost."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

sys.path.insert(0, r"C:\Users\mvk80\Documents\Codex\2026-11-05\search-lacuna\work\FMG-research\sensitivity")
from run_typeaware_imputation import load_datasets

SEEDS = [42, 123, 456, 1, 7]


def encode_train_test(Xt, Xv):
    out_t, out_v = [], []
    for c in Xt.columns:
        if Xt[c].dtype == object or str(Xt[c].dtype).startswith("category"):
            cats = pd.Index(Xt[c].dropna().astype(str).unique())
            lookup = {k: i for i, k in enumerate(cats)}
            out_t.append(Xt[c].astype(str).map(lookup).fillna(-1).to_numpy(float))
            out_v.append(Xv[c].astype(str).map(lookup).fillna(-1).to_numpy(float))
        else:
            out_t.append(pd.to_numeric(Xt[c], errors="coerce").to_numpy(float))
            out_v.append(pd.to_numeric(Xv[c], errors="coerce").to_numpy(float))
    return np.column_stack(out_t), np.column_stack(out_v)


def mask(X, rate, seed):
    rng = np.random.RandomState(seed)
    m = rng.rand(*X.shape) < rate
    out = X.copy()
    out[m] = np.nan
    return out


def score(model, X, y, g):
    p = model.predict(X)
    g0, g1 = g == 0, g == 1
    return (float(accuracy_score(y, p)), float(recall_score(y[g0], p[g0], zero_division=0)),
            float(recall_score(y[g1], p[g1], zero_division=0)))


def main():
    out = Path("native_routing_pilot_outputs").resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dname, (df, features, sensitive, target, declared_cat) in load_datasets().items():
        print("starting", dname, flush=True)
        X, y, g = df[features].copy(), df[target].to_numpy(int), df[sensitive].to_numpy(int)
        for seed in SEEDS:
            Xt, Xv, yt, yv, gt, gv = train_test_split(X, y, g, test_size=.30, random_state=seed, stratify=y)
            A, B = encode_train_test(Xt, Xv)
            Bm = mask(B, .20, seed)
            for name, model in [
                ("LightGBM", LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=.05,
                                              random_state=seed, verbosity=-1)),
                ("XGBoost", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.05,
                                            subsample=.8, colsample_bytree=.8, random_state=seed,
                                            eval_metric="logloss", n_jobs=4, tree_method="hist")),
            ]:
                model.fit(A, yt)
                clean = score(model, B, yv, gv); masked = score(model, Bm, yv, gv)
                ad = clean[0] - masked[0]
                rows.append({"dataset": dname, "model": name, "seed": seed, "mask_rate": .20,
                             "accuracy_drop": ad, "recall_drop_g0": clean[1]-masked[1],
                             "fmg_g0": clean[1]-masked[1]-ad, "recall_drop_g1": clean[2]-masked[2],
                             "fmg_g1": clean[2]-masked[2]-ad})
        pd.DataFrame(rows).to_csv(out / "native_routing_pilot_per_seed.csv", index=False, float_format="%.8f")
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "native_routing_pilot_per_seed.csv", index=False, float_format="%.8f")
    summary = raw.groupby(["dataset", "model"], as_index=False).agg(
        n_seeds=("fmg_g0", "size"), mean_fmg_g0=("fmg_g0", "mean"), sd_fmg_g0=("fmg_g0", "std"),
        positive_fmg_g0_seeds=("fmg_g0", lambda x: int((x > 0).sum())),
        mean_accuracy_drop=("accuracy_drop", "mean"), mean_recall_drop_g0=("recall_drop_g0", "mean"),
        mean_fmg_g1=("fmg_g1", "mean"), positive_fmg_g1_seeds=("fmg_g1", lambda x: int((x > 0).sum())),
    )
    summary.to_csv(out / "native_routing_pilot_summary.csv", index=False, float_format="%.8f")
    (out / "native_routing_pilot_manifest.txt").write_text(
        "Five-seed, 20% MCAR native-NaN routing pilot. Models: LightGBM 4.7.0 and XGBoost 3.4.1.\n"
        "Categorical features are train-fitted ordinal encoded only to provide numeric model input; masked test cells remain NaN.\n"
        "Supplemental pilot; not a matched reproduction of the primary four-model protocol.\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
