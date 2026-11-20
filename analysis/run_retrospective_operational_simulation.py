"""Retrospective alert-rule simulation from exported row-level predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def recall(g):
    return float(((g.y == 1) & (g.pred == 1)).sum() / max((g.y == 1).sum(), 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', default='retrospective_operational_outputs')
    ap.add_argument('--threshold', type=float, default=0.01)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = {'dataset','model','seed','condition','subgroup','y','pred'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    rows = []
    for keys, g in df.groupby(['dataset','model','seed'], sort=True):
        clean = g[g.condition == 'complete']; masked = g[g.condition.str.startswith('masked_')]
        if len(clean) == 0 or len(masked) == 0: continue
        clean_acc = float((clean.y == clean.pred).mean()); masked_acc = float((masked.y == masked.pred).mean())
        f0c = recall(clean[clean.subgroup == 0]); f0m = recall(masked[masked.subgroup == 0])
        c1c = recall(clean[clean.subgroup == 1]); c1m = recall(masked[masked.subgroup == 1])
        accuracy_loss = clean_acc - masked_acc; focus_loss = f0c - f0m
        row = {'dataset':keys[0], 'model':keys[1], 'seed':int(keys[2]),
               'accuracy_loss':accuracy_loss, 'focus_recall_loss':focus_loss,
               'comparison_recall_loss':c1c-c1m, 'fmg':focus_loss-accuracy_loss,
               'hidden_event':bool(focus_loss >= args.threshold and accuracy_loss < args.threshold),
               'accuracy_only_alert':bool(accuracy_loss >= args.threshold),
               'fmg_supported_alert':bool(focus_loss-accuracy_loss >= args.threshold)}
        rows.append(row)
    cell = pd.DataFrame(rows)
    cell.to_csv(out/'operational_cell_results.csv', index=False, float_format='%.8f')
    records=[]
    for keys,g in cell.groupby(['dataset','model'], sort=True):
        h=g.hidden_event
        records.append({'dataset':keys[0], 'model':keys[1], 'n_cells':len(g),
          'hidden_events':int(h.sum()), 'accuracy_only_detected_hidden':int((h & g.accuracy_only_alert).sum()),
          'fmg_detected_hidden':int((h & g.fmg_supported_alert).sum()),
          'accuracy_only_alerts':int(g.accuracy_only_alert.sum()), 'fmg_supported_alerts':int(g.fmg_supported_alert.sum()),
          'accuracy_only_false_alerts':int((g.accuracy_only_alert & ~h).sum()),
          'fmg_supported_false_alerts':int((g.fmg_supported_alert & ~h).sum()),
          'accuracy_only_hidden_detection_rate':float((h & g.accuracy_only_alert).sum()/max(h.sum(),1)),
          'fmg_hidden_detection_rate':float((h & g.fmg_supported_alert).sum()/max(h.sum(),1))})
    summary=pd.DataFrame(records)
    summary.to_csv(out/'operational_rule_summary.csv', index=False, float_format='%.8f')
    overall=pd.DataFrame([{'threshold':args.threshold, 'n_cells':len(cell), 'hidden_events':int(cell.hidden_event.sum()),
      'accuracy_only_detected_hidden':int((cell.hidden_event & cell.accuracy_only_alert).sum()),
      'fmg_detected_hidden':int((cell.hidden_event & cell.fmg_supported_alert).sum()),
      'accuracy_only_alerts':int(cell.accuracy_only_alert.sum()), 'fmg_supported_alerts':int(cell.fmg_supported_alert.sum()),
      'accuracy_only_false_alerts':int((cell.accuracy_only_alert & ~cell.hidden_event).sum()),
      'fmg_supported_false_alerts':int((cell.fmg_supported_alert & ~cell.hidden_event).sum())}])
    overall.to_csv(out/'operational_overall_summary.csv', index=False, float_format='%.8f')
    (out/'operational_manifest.txt').write_text(
      'Retrospective operational screening simulation using exported row-level predictions. '
      'Hidden event = focus-group recall loss >= threshold and aggregate accuracy loss < threshold. '
      'Accuracy-only alert = accuracy loss >= threshold. FMG-supported alert = FMG >= threshold. '
      'This is not a human-in-the-loop, prospective deployment, clinical validation, or proof of operational benefit.\n', encoding='utf-8')
    print(overall.to_string(index=False))

if __name__ == '__main__': main()
