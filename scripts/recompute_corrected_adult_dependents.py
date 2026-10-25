"""Recompute Adult-dependent descriptive sensitivities without touching archives."""
from pathlib import Path
import zipfile
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"adult_corrected_rerun"
raw=pd.read_csv(OUT/"results_raw_10seeds_with_corrected_adult.csv")
primary_ds={"Adult Income","COMPAS","Taiwan Credit"}
models={"RF","XGB","LGB","MLP"}
seeds=[42,123,456,1,7,13,21,37,55,99]

# Complementary-group FMG directly compares the same loss construction for g1.
sel=raw[raw.dataset.isin(primary_ds)&raw.model.isin(models)&raw.rate.isin([0.0,.20])]
wide=sel.pivot(index=["dataset","model","seed"],columns="rate",values=["accuracy","recall_g0","recall_g1"])
wide.columns=[f"{metric}_{int(rate*100)}" for metric,rate in wide.columns]
wide=wide.reset_index()
wide["accuracy_drop"]=wide.accuracy_0-wide.accuracy_20
wide["fmg_g0"]=wide.recall_g0_0-wide.recall_g0_20-wide.accuracy_drop
wide["fmg_g1"]=wide.recall_g1_0-wide.recall_g1_20-wide.accuracy_drop
wide.to_csv(OUT/"complementary_group_per_seed_corrected_adult.csv",index=False,float_format="%.8f")
comp=wide.groupby(["dataset","model"],as_index=False).agg(
    n_seeds=("fmg_g1","size"),mean_fmg_g0=("fmg_g0","mean"),mean_fmg_g1=("fmg_g1","mean"),
    sd_fmg_g1=("fmg_g1","std"),positive_fmg_g1_seeds=("fmg_g1",lambda x:int((x>0).sum())))
comp.to_csv(OUT/"complementary_group_summary_corrected_adult.csv",index=False,float_format="%.8f")

# Baseline recall versus primary FMG Spearman summaries.
base=sel[sel.rate.eq(0)][["dataset","model","seed","recall_g0"]].rename(columns={"recall_g0":"clean_baseline_recall_g0"})
stress=sel[sel.rate.eq(.20)][["dataset","model","seed","masking_gap","accuracy","recall_g0"]].rename(columns={"masking_gap":"fmg_at_20pct_mcar","accuracy":"masked_accuracy","recall_g0":"masked_recall_g0"})
paired=base.merge(stress,on=["dataset","model","seed"],validate="one_to_one")
paired.to_csv(OUT/"baseline_recall_per_seed_corrected_adult.csv",index=False,float_format="%.8f")
baseline=[]
for (ds,model),g in paired.groupby(["dataset","model"],sort=False):
    rho=g.clean_baseline_recall_g0.rank(method="average").corr(g.fmg_at_20pct_mcar.rank(method="average"))
    baseline.append({"dataset":ds,"model":model,"n_seed_pairs":len(g),"baseline_recall_min":g.clean_baseline_recall_g0.min(),"baseline_recall_max":g.clean_baseline_recall_g0.max(),"spearman_rho_baseline_recall_vs_fmg":rho})
pd.DataFrame(baseline).to_csv(OUT/"baseline_recall_summary_corrected_adult.csv",index=False,float_format="%.8f")

# Threshold-discordance grid for all three primary datasets, four models,
# ten paired splits, three missingness rates and four descriptive thresholds.
taus=[.01,.025,.05,.10]
threshold_rows=[]
for ds in primary_ds:
  for model in models:
    d=raw[(raw.dataset.eq(ds))&raw.model.eq(model)]
    clean=d[d.rate.eq(0)].set_index("seed")
    for rate in [.05,.10,.20]:
      masked=d[np.isclose(d.rate,rate)].set_index("seed")
      for seed in seeds:
        acc_loss=float(clean.loc[seed,"accuracy"]-masked.loc[seed,"accuracy"])
        recall_loss=float(clean.loc[seed,"recall_g0"]-masked.loc[seed,"recall_g0"])
        fmg=recall_loss-acc_loss
        for delta in taus:
          rflag=recall_loss>=delta; aflag=acc_loss>=delta
          threshold_rows.append({"dataset":ds,"model":model,"seed":seed,"rate":rate,"threshold":delta,
                                 "accuracy_loss":acc_loss,"g0_recall_loss":recall_loss,"fmg":fmg,
                                 "recall_flag":int(rflag),"accuracy_flag":int(aflag),
                                 "discordant_recall_only":int(rflag and not aflag)})
th=pd.DataFrame(threshold_rows)
th.to_csv(OUT/"diagnostic_discordance_per_seed_corrected_adult.csv",index=False,float_format="%.8f")
summary=th.groupby(["rate","threshold"],as_index=False).agg(
  n=("seed","size"),recall_flags=("recall_flag","sum"),accuracy_flags=("accuracy_flag","sum"),
  recall_only_flags=("discordant_recall_only","sum"))
summary.to_csv(OUT/"diagnostic_discordance_summary_corrected_adult.csv",index=False)

# Corrected Adult positive-class prevalence contrast under the exact parser.
archive=Path(r"C:\Users\mvk80\Downloads\adult.zip")
cols=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
with zipfile.ZipFile(archive) as z:
    member=next(n for n in z.namelist() if n.lower().endswith("adult.data"))
    adult=pd.read_csv(z.open(member),header=None,names=cols,na_values=["?"," ?"],skipinitialspace=True).dropna()
adult["label"]=(adult.income.str.strip().str.replace(".","",regex=False)==">50K").astype(int)
rates=adult.groupby("sex").label.mean()
gap=float(rates["Male"]-rates["Female"])
print("Corrected Adult male-female positive-rate difference:",gap)
print("Threshold grid at 20%:")
print(summary[np.isclose(summary.rate,.20)].to_string(index=False))
print("Complementary-group summaries:")
print(comp.to_string(index=False))
print("Baseline Spearman range:",float(pd.DataFrame(baseline).spearman_rho_baseline_recall_vs_fmg.min()),float(pd.DataFrame(baseline).spearman_rho_baseline_recall_vs_fmg.max()))
