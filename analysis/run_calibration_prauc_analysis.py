from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Compute calibration and PR-AUC sensitivity from row-level probabilities.')
parser.add_argument('input_csv', help='CSV containing clean_score, score, y, subgroup, dataset, model, seed, and method.')
parser.add_argument('--output-dir', default='calibration_prauc_results')
args = parser.parse_args()
INPUT = Path(args.input_csv)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(exist_ok=True)

def ece_equal_width(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    value = 0.0
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b+1]
        sel = (p >= lo) & ((p < hi) if b < n_bins - 1 else (p <= hi))
        n = int(sel.sum())
        if n:
            acc = float(y[sel].mean())
            conf = float(p[sel].mean())
            gap = abs(acc - conf)
            value += (n / total) * gap
            rows.append({'bin': b, 'lower': lo, 'upper': hi, 'n': n,
                         'empirical_rate': acc, 'mean_score': conf, 'gap': gap})
        else:
            rows.append({'bin': b, 'lower': lo, 'upper': hi, 'n': 0,
                         'empirical_rate': np.nan, 'mean_score': np.nan, 'gap': np.nan})
    return float(value), rows

def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    vals = {'brier': float(np.mean((p - y) ** 2))}
    if len(np.unique(y)) == 2:
        order = np.argsort(-p, kind='mergesort')
        ys = y[order]
        tp = np.cumsum(ys)
        ranks = np.arange(1, len(ys) + 1)
        precision = tp / ranks
        vals['pr_auc'] = float(np.sum(precision[ys == 1]) / np.sum(ys))
    else:
        vals['pr_auc'] = np.nan
    vals['ece10'], bins = ece_equal_width(y, p, 10)
    return vals, bins

df = pd.read_csv(INPUT)
required = {'dataset','model','seed','mask_rate','subgroup','y','clean_score','score','method'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')

rows = []
bin_rows = []
keys = ['dataset','model','seed','mask_rate','method']
for key, g in df.groupby(keys, sort=True):
    for subgroup_name, h in [('all', g), ('focus_g0', g[g['subgroup'] == 0])]:
        if h.empty or h['y'].nunique() < 2:
            continue
        for condition, score_col in [('clean', 'clean_score'), ('masked', 'score')]:
            vals, bins = metrics(h['y'], h[score_col])
            row = dict(zip(keys, key))
            row.update({'subgroup_scope': subgroup_name, 'condition': condition,
                        'n': len(h), 'positive_n': int(h.y.sum()), **vals})
            rows.append(row)
            for b in bins:
                br = dict(zip(keys, key))
                br.update({'subgroup_scope': subgroup_name, 'condition': condition, **b})
                bin_rows.append(br)

per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT_DIR / 'calibration_prauc_per_seed.csv', index=False)
pd.DataFrame(bin_rows).to_csv(OUT_DIR / 'calibration_ece_bins.csv', index=False)

pivot = per_seed.pivot_table(index=keys + ['subgroup_scope'], columns='condition',
                             values=['brier','ece10','pr_auc'])
pivot.columns = [f'{a}_{b}' for a, b in pivot.columns]
pivot = pivot.reset_index()
for metric in ['brier','ece10','pr_auc']:
    pivot[f'{metric}_change_masked_minus_clean'] = pivot[f'{metric}_masked'] - pivot[f'{metric}_clean']
pivot.to_csv(OUT_DIR / 'calibration_prauc_paired_changes.csv', index=False)

summary = pivot.groupby(['dataset','model','method','subgroup_scope'], as_index=False).agg(
    seeds=('seed','nunique'),
    brier_clean=('brier_clean','mean'), brier_masked=('brier_masked','mean'),
    brier_change=('brier_change_masked_minus_clean','mean'),
    ece10_clean=('ece10_clean','mean'), ece10_masked=('ece10_masked','mean'),
    ece10_change=('ece10_change_masked_minus_clean','mean'),
    pr_auc_clean=('pr_auc_clean','mean'), pr_auc_masked=('pr_auc_masked','mean'),
    pr_auc_change=('pr_auc_change_masked_minus_clean','mean'))
summary.to_csv(OUT_DIR / 'calibration_prauc_summary.csv', index=False)

manifest = {
    'input': str(INPUT), 'mask_rate': 0.20, 'conditions': ['clean','masked'],
    'methods': sorted(df.method.dropna().unique().tolist()),
    'models': sorted(df.model.dropna().unique().tolist()),
    'datasets': sorted(df.dataset.dropna().unique().tolist()),
    'seeds': sorted(df.seed.dropna().unique().tolist()),
    'metrics': {'brier': 'brier_score_loss',
                'ece10': 'equal-width 10-bin expected calibration error',
                'pr_auc': 'average_precision_score (average precision / PR-AUC)'},
    'subgroup_scopes': ['all','focus_g0'],
    'note': 'Computed from existing row-level probability outputs; no model retraining.'
}
(OUT_DIR / 'calibration_prauc_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(summary.to_string(index=False))
print('Wrote', OUT_DIR)
