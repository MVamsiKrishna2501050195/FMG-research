"""Regenerate manuscript tables/figures using corrected Adult rows.

This deliberately combines the Adult-only corrected rerun with unchanged
archived non-Adult rows. It does not claim to rerun the full primary pipeline.
All outputs are written under adult_corrected_rerun/.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "adult_corrected_rerun"
FIG = OUT / "corrected_figures"
FIG.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(OUT / "results_raw_10seeds_with_corrected_adult.csv")
table3 = pd.read_csv(OUT / "table3_primary_components_corrected.csv")
mar = pd.read_csv(OUT / "group_conditional_mar_per_seed_with_corrected_adult.csv")
matched = pd.read_csv(OUT / "adult_corrected_rerun" / "adult_corrected_matched_subsample_per_seed.csv") if (OUT / "adult_corrected_rerun" / "adult_corrected_matched_subsample_per_seed.csv").exists() else pd.read_csv(OUT / "adult_corrected_matched_subsample_per_seed.csv")
typeaware_old = pd.read_csv(OUT / "typeaware_imputation_summary_with_corrected_adult.csv")
typeaware_adult_rerun = pd.read_csv(
    OUT / "typeaware_imputation_outputs" / "typeaware_imputation_sensitivity_summary.csv"
)
typeaware = pd.concat([
    typeaware_old[typeaware_old.dataset != "Adult Income"],
    typeaware_adult_rerun,
], ignore_index=True)

SEEDS = [42, 123, 456, 1, 7, 13, 21, 37, 55, 99]
MODELS = ["LR", "RF", "XGB", "LGB", "MLP"]
PRIMARY = ["Adult Income", "COMPAS", "Taiwan Credit"]
COLORS = {"LR":"#2196F3", "RF":"#F44336", "XGB":"#4CAF50", "LGB":"#9C27B0", "MLP":"#FF9800"}

def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

# Fig. 3 in the manuscript: primary MCAR trajectories, mean +/- 1 SD.
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5), sharex=True)
for ax, ds in zip(axes.flat, ["Adult Income", "COMPAS", "Taiwan Credit", "German Credit"]):
    d = raw[raw.dataset.eq(ds)]
    for model in MODELS:
        s = d[d.model.eq(model)].groupby("rate").masking_gap.agg(["mean", "std"]).reindex([0, .05, .10, .20])
        ax.errorbar(s.index * 100, s["mean"], yerr=s["std"], marker="o", capsize=2,
                    linewidth=1.5, markersize=4, color=COLORS[model], label=model)
    ax.axhline(0, color="#555", linewidth=.8)
    ax.set_title(ds + (" (exploratory)" if ds == "German Credit" else ""))
    ax.set_ylabel("FMG")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=7, ncol=2)
for ax in axes[-1]: ax.set_xlabel("Test-feature cells masked (%)")
fig.suptitle("Primary frozen-classifier audit: MCAR masking-gap trajectories\nMean +/- 1 SD across ten repeated splits")
fig.tight_layout(rect=[0, 0, 1, .94])
save(fig, "fig03_masking_gap_curves_corrected_adult")

# Fig. 4: accuracy and subgroup-recall loss at 20% MCAR, three primary datasets.
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)
for ax, ds in zip(axes, PRIMARY):
    d = raw[(raw.dataset.eq(ds)) & raw.model.isin(["RF", "XGB", "LGB", "MLP"])]
    clean = d[d.rate.eq(0)].set_index(["model", "seed"])
    stress = d[d.rate.eq(.20)].set_index(["model", "seed"])
    x = np.arange(4); w = .34
    for j, (metric, color, label) in enumerate([("accuracy", "#4F95D5", "Accuracy loss"), ("recall_g0", "#E56B78", "g0 recall loss")]):
        vals=[]; errs=[]
        for model in ["RF", "XGB", "LGB", "MLP"]:
            vals_s = (clean.loc[model][metric] - stress.loc[model][metric]).reindex(SEEDS)
            vals.append(vals_s.mean()); errs.append(vals_s.std(ddof=1))
        ax.bar(x + (j-.5)*w, vals, w, yerr=errs, capsize=3, color=color, label=label, alpha=.9)
    ax.set_xticks(x, ["RF", "XGB", "LGB", "MLP"])
    ax.set_title(ds); ax.set_xlabel("Classifier"); ax.set_ylabel("Loss at 20% MCAR")
    ax.axhline(0, color="#555", linewidth=.8); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
fig.suptitle("Aggregate accuracy and focus-group recall loss at 20% MCAR\nMean +/- 1 SD across ten repeated splits")
fig.tight_layout(rect=[0, 0, 1, .92])
save(fig, "fig04_accuracy_vs_recall_corrected_adult")

# Fig. 5: group-conditional MAR-like paired contrasts from actual per-seed rows.
fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), sharey=True)
for ax, rate in zip(axes, [.10, .20]):
    d=mar[np.isclose(mar.rate_nominal, rate)]
    labels=[]; y=[]
    for di, ds in enumerate(PRIMARY):
        for mi, model in enumerate(["RF", "XGB", "LGB", "MLP"]):
            sub=d[d.dataset.eq(ds)&d.model.eq(model)].sort_values("seed", key=lambda s:s.map({v:i for i,v in enumerate(SEEDS)}))
            dif=sub.mar_minus_mcar_mg.to_numpy()
            yy=di*4.7+mi
            ax.errorbar(dif.mean(), yy, xerr=dif.std(ddof=1), fmt="o", capsize=3,
                        color={"Adult Income":"#1976D2","COMPAS":"#E65100","Taiwan Credit":"#009688"}[ds])
            labels.append(f"{ds} - {model}"); y.append(yy)
        if di<2: ax.axhline((di+1)*4.7-.35,color="#bbb",lw=.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.axvline(0,color="#333",lw=.8)
    ax.set_title(f"{int(rate*100)}% nominal masking"); ax.set_xlabel("FMG change: constructed MAR-like minus MCAR")
    ax.grid(axis="x",alpha=.25)
axes[0].set_ylabel("Dataset and classifier")
fig.suptitle("Group-conditional missingness sensitivity\nPaired mean contrast +/- 1 SD across ten splits")
fig.tight_layout(rect=[.12,0,1,.93])
save(fig,"fig05_group_conditional_vs_mcar_corrected_adult")

# Fig. 6: corrected Adult matched-subsample line plot. The right panel retains
# the original descriptive class-rate points from the archived benchmark.
orig=raw[(raw.dataset.eq("Adult Income"))&raw.model.eq("RF")&np.isclose(raw.rate,.20)].set_index("seed").reindex(SEEDS)
mat=matched.set_index("seed").reindex(SEEDS)
fig, axes=plt.subplots(1,2,figsize=(12.6,4.8))
ax=axes[0]; x=np.arange(len(SEEDS))
ax.plot(x,orig.masking_gap,"o-",color="#E06C75",label=f"Original (mean={orig.masking_gap.mean():.3f})")
ax.plot(x,mat.masking_gap,"s--",color="#5B9BD5",label=f"Matched sample (mean={mat.masking_gap.mean():.3f})")
ax.axhline(0,color="#333",lw=.8); ax.set_xticks(x,[str(s) for s in SEEDS],rotation=30); ax.set_xlabel("Repeated split seed"); ax.set_ylabel("FMG at 20% MCAR"); ax.set_title("Adult Income RF: original vs matched"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.25)
ax=axes[1]
for ds,color in zip(PRIMARY,["#E56B78","#4F95D5","#4CAF50"]):
    d=table3[(table3.dataset.eq(ds))&table3.model.eq("RF")].iloc[0]
    gap={"Adult Income":.200,"COMPAS":-.126,"Taiwan Credit":.034}[ds]
    ax.scatter(gap,d.fmg_mean,s=90,color=color,label=f"{ds} (class-rate diff={gap:+.3f})")
ax.axvline(0,color="#aaa",ls="--",lw=.8); ax.axhline(0,color="#333",lw=.8); ax.set_xlabel("Positive-class rate difference (g1 - g0)"); ax.set_ylabel("RF mean FMG at 20% MCAR"); ax.set_title("Descriptive association (not causal)"); ax.legend(fontsize=7); ax.grid(alpha=.2)
fig.suptitle("Matched-subsample and class-rate descriptive sensitivity")
fig.tight_layout(rect=[0,0,1,.92]); save(fig,"fig06_class_imbalance_corrected_adult")

# Fig. 8: preserve the unaffected COMPAS-only SHAP panel. Adult SHAP values
# depend on the corrected sample and are intentionally not carried forward.
shap_csv = ROOT / "shap_subgroup_results.csv"
if shap_csv.exists():
    sh=pd.read_csv(shap_csv); sh=sh[sh.dataset.eq("COMPAS")]
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    if not sh.empty:
        asym=sh.pivot(index="feature",columns="subgroup",values="shap_shift")
        asym["asymmetry"]=abs(asym.get(0,0)-asym.get(1,0))
        feats=asym.nlargest(6,"asymmetry").index.tolist()
        z=sh[sh.feature.isin(feats)]
        pos={f:i for i,f in enumerate(feats)}; h=.34
        for _,row in z.iterrows():
            ax.barh(pos[row.feature]+(h/2 if row.subgroup==0 else -h/2),row.shap_shift,h,
                    color="#E06C75" if row.subgroup==0 else "#82B0D2",
                    label="g0 (focus)" if row.subgroup==0 and pos[row.feature]==0 else "g1 (comparison)" if row.subgroup==1 and pos[row.feature]==0 else "")
        ax.set_yticks(range(len(feats)),feats); ax.axvline(0,color="#333",lw=.8)
        ax.set_xlabel("SHAP shift (masked - clean)"); ax.legend(fontsize=8); ax.grid(axis="x",alpha=.25)
    ax.set_title("COMPAS RF, seed 42: exploratory SHAP shifts at 20% MCAR")
    fig.tight_layout(); save(fig,"fig08_compas_shap_only")

# Fig. 9: plot updated multi-seed, type-aware sensitivity instead of stale
# single-seed encoded-space values.
fig,axes=plt.subplots(1,3,figsize=(12.4,4.2),sharey=True)
strats=[("median_mode","Median/mode"),("knn_numeric_mode","KNN/mode"),("iterative_numeric_mode","Iterative/mode")]
datasets=["Adult Income","COMPAS","Taiwan Credit"]
palette=["#5B9BD5","#E06C75","#4CAF50"]
for ax,ds in zip(axes,datasets):
    d=typeaware[typeaware.dataset.eq(ds)].set_index("imputation_strategy")
    for i,(key,label) in enumerate(strats):
        if key not in d.index: continue
        row=d.loc[key]
        ax.bar(i,row.mean_fmg_g0,yerr=row.sd_fmg_g0,capsize=3,color=palette[i],alpha=.9)
        ax.text(i,row.mean_fmg_g0+(0.002 if row.mean_fmg_g0>=0 else -0.004),f"{int(row.positive_fmg_g0_seeds)}/10 +",ha="center",va="bottom" if row.mean_fmg_g0>=0 else "top",fontsize=7)
    ax.axhline(0,color="#333",lw=.8); ax.set_xticks(range(3),["Median/\nmode","KNN/\nmode","Iterative/\nmode"]); ax.set_title(ds); ax.grid(axis="y",alpha=.25)
axes[0].set_ylabel("Mean focus-group FMG at 20% MCAR (+/- 1 SD)")
fig.suptitle("Type-aware imputation sensitivity across ten repeated splits (RF)")
fig.tight_layout(rect=[0,0,1,.91]); save(fig,"fig09_type_aware_imputation_corrected_adult")

# Compact manifest for figure lineage and the Adult input correction.
manifest={"scope":"Adult-only corrected primary rerun merged with unchanged archived non-Adult rows; not a full pipeline rerun",
          "adult_input_sha256":json.load(open(OUT/"run_manifest.json",encoding="utf8"))["input_sha256"],
          "figures":[p.name for p in sorted(FIG.glob("*.png"))]}
(FIG/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
# Publish the regenerated figure assets at the repository's canonical paths.
canonical = {
    "fig03_masking_gap_curves_corrected_adult": "fig03_masking_gap_curves",
    "fig04_accuracy_vs_recall_corrected_adult": "fig04_accuracy_vs_recall",
    "fig05_group_conditional_vs_mcar_corrected_adult": "fig05_group_conditional_vs_mcar",
    "fig06_class_imbalance_corrected_adult": "fig06_class_imbalance",
    "fig08_compas_shap_only": "fig08_shap_shifts",
    "fig09_type_aware_imputation_corrected_adult": "fig09_imputation_ablation",
}
for src_stem,dst_stem in canonical.items():
    for ext in ("png","pdf"):
        src=FIG/f"{src_stem}.{ext}"
        if src.exists(): shutil.copy2(src,ROOT/"figures"/f"{dst_stem}.{ext}")
print(json.dumps(manifest,indent=2))
