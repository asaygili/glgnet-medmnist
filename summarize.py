"""Aggregate results/runs.jsonl -> mean±std AUC/ACC per (model, dataset). Usage: python summarize.py [tag ...]"""
import sys
import pandas as pd

df = pd.read_json('results/runs.jsonl', lines=True)
df = df[df.tag != 'smoke']
if len(sys.argv) > 1:
    df = df[df.tag.isin(sys.argv[1:])]
g = df.groupby(['data', 'model'])
out = pd.DataFrame({
    'n': g.size(),
    'AUC': g.test_auc.mean().map('{:.3f}'.format) + '±' + g.test_auc.std().fillna(0).map('{:.3f}'.format),
    'ACC': g.test_acc.mean().map('{:.3f}'.format) + '±' + g.test_acc.std().fillna(0).map('{:.3f}'.format),
    'Mparams': (g.params.mean() / 1e6).round(2),
    'time_s': g.train_time_s.mean().round(0),
})
print(out.to_string())
