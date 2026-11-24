"""Results tables in paper/tables/ from results/*.  Usage: python make_results_tables.py (rerun any time).
Comparison tables use seeds 0-2 for every model; models whose runs are incomplete are left out automatically."""
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata, friedmanchisquare
from medmnist import INFO

df = pd.DataFrame([json.loads(l) for l in open('results/runs.jsonl')])
df = df[(df.tag == 'main') & (df.seed < 3)]
G = df.groupby(['size', 'data', 'model'])[['test_auc', 'test_acc']]
MU, SD, CNT = G.mean(), G.std(), G.size()
ALLD = ['pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist',
        'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
SMALL = ['dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'organcmnist', 'organsmnist']
NAME = {'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50', 'mobilenetv2': 'MobileNetV2', 'shufflenetv2': 'ShuffleNetV2',
        'efficientnetb0': 'EfficientNet-B0', 'vit-tiny': 'ViT-Ti', 'swin-t': 'Swin-T', 'mobilevit-s': 'MobileViT-S',
        'glgnet': 'GLGNet', 'glgnet-sum': 'Sum', 'glgnet-local': 'Local only', 'glgnet-global': 'Global only'}
PARAMS = df[df['size'] == 28].groupby('model').params.median() / 1e6
M = {'test_auc': 'AUC', 'test_acc': 'ACC'}
# published MedMNIST v2 benchmark (Yang et al. 2023, Table 3), 28x28: (AUC, ACC)
OFFICIAL = {'resnet18': [(.983, .907), (.768, .947), (.917, .735), (.943, .743), (.944, .854), (.717, .524), (.901, .863),
                         (.998, .958), (.930, .676), (.997, .935), (.992, .900), (.972, .782)],
            'resnet50': [(.990, .911), (.769, .947), (.913, .735), (.952, .762), (.948, .854), (.726, .528), (.857, .812),
                         (.997, .956), (.931, .680), (.997, .935), (.992, .905), (.972, .770)]}


def complete(z, datasets, m):
    return all(CNT.get((z, d, m), 0) == 3 for d in datasets)


def mu(z, d, m, k):
    return MU.loc[(z, d, m), k]


def cell(v, best, sd=None):
    s = f'{v:.3f}'
    s = f'\\textbf{{{s}}}' if round(v, 3) == best else s
    return s + (f'{{\\tiny$\\pm${sd:.3f}}}' if sd is not None else '')


def pfmt(p):
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def holm(ps):
    ps = np.asarray(ps); adj = np.empty(len(ps)); run = 0
    for r, j in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - r) * ps[j]); adj[j] = min(1, run)
    return adj


def write(name, body):  # shrink wide tables to the text width
    body = body.replace(r'\begin{tabular}', r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}').replace(r'\end{tabular}', r'\end{tabular}\end{adjustbox}')
    with open(f'paper/tables/{name}.tex', 'w', encoding='utf-8') as f:
        f.write(body)


def per_metric_table(z, datasets, models, k, ref, caption, label, env='table*'):
    """rows = datasets, cols = models; Average, mean rank, Holm-adjusted Wilcoxon p vs `ref` over datasets."""
    models = [m for m in models if complete(z, datasets, m)]
    lines = []
    for d in datasets:
        vals = [mu(z, d, m, k) for m in models]
        best = round(max(vals), 3)
        lines.append(INFO[d]['python_class'] + ' & ' +
                     ' & '.join(cell(v, best, SD.loc[(z, d, m), k]) for v, m in zip(vals, models)) + ' \\\\')
    A = np.array([[mu(z, d, m, k) for m in models] for d in datasets])
    avg = A.mean(0)
    best = round(avg.max(), 3)
    ranks = np.mean([rankdata(-r) for r in A], 0)
    r = models.index(ref)
    others = [j for j in range(len(models)) if j != r]
    raw = [wilcoxon(A[:, r], A[:, j]).pvalue for j in others]
    adj = dict(zip(others, holm(raw)))
    print(f'{label}: Friedman p = {friedmanchisquare(*A.T).pvalue:.2g}; raw p vs {ref}:',
          {models[j]: round(p, 4) for j, p in zip(others, raw)}, '; Holm:', {models[j]: round(adj[j], 4) for j in others})
    lines += ['\\midrule', 'Average & ' + ' & '.join(cell(v, best) for v in avg) + ' \\\\',
              'Mean rank $\\downarrow$ & ' + ' & '.join(f'{v:.2f}' for v in ranks) + ' \\\\',
              f'$p$ vs {NAME[ref]} & ' + ' & '.join('--' if j == r else pfmt(adj[j]) for j in range(len(models))) + ' \\\\']
    head = ' & '.join(NAME[m] for m in models)
    par = ' & '.join(f'{PARAMS[m]:.1f}\\,M' for m in models)
    write(label, f'\\begin{{{env}}}[t]\\centering\\footnotesize\n\\caption{{{caption}}}\\label{{tab:{label}}}\n'
          f'\\begin{{tabular}}{{l{"c" * len(models)}}}\\toprule\nDataset & {head} \\\\\nParams & {par} \\\\\\midrule\n'
          + '\n'.join(lines) + f'\n\\bottomrule\\end{{tabular}}\n\\end{{{env}}}\n')


BASE = ['resnet18', 'resnet50', 'mobilenetv2', 'shufflenetv2', 'efficientnetb0', 'vit-tiny', 'swin-t', 'mobilevit-s', 'glgnet']
for k in M:
    per_metric_table(28, ALLD, BASE, k, 'glgnet',
                     f'Test {M[k]} on the twelve MedMNIST v2 datasets at $28\\times28$ (mean{{\\tiny$\\pm$std}} over three seeds). '
                     'Best per row in bold. Mean ranks and two-sided Wilcoxon signed-rank $p$-values against GLGNet '
                     '(Holm-adjusted) are computed over the twelve per-dataset means.', f'main_{M[k].lower()}')
    per_metric_table(28, SMALL, ['glgnet', 'glgnet-sum', 'glgnet-local', 'glgnet-global'], k, 'glgnet',
                     f'Ablation of the fusion mechanism at $28\\times28$: test {M[k]} (mean{{\\tiny$\\pm$std}} over three seeds; '
                     '$p$-values Holm-adjusted).', f'ablation_{M[k].lower()}', env='table')

# reproduction check against the published benchmark
lines = []
for i, d in enumerate(ALLD):
    lines.append(INFO[d]['python_class'] + ' & ' + ' & '.join(
        f'{OFFICIAL[m][i][j]:.3f} & {mu(28, d, m, k):.3f}' for m in ('resnet18', 'resnet50') for j, k in enumerate(M)) + ' \\\\')
avg = [f'{np.mean([OFFICIAL[m][i][j] for i in range(12)]):.3f} & {np.mean([mu(28, d, m, k) for d in ALLD]):.3f}'
       for m in ('resnet18', 'resnet50') for j, k in enumerate(M)]
write('repro', '\\begin{table}[t]\\centering\\footnotesize\n'
      '\\caption{Reproduction check at $28\\times28$: results published with MedMNIST v2~\\cite{yang2023medmnistv2} (Pub.) and our '
      're-training with the same protocol (Ours, mean of three seeds).}\\label{tab:repro}\n'
      '\\begin{tabular}{lcccccccc}\\toprule\n& \\multicolumn{4}{c}{ResNet-18} & \\multicolumn{4}{c}{ResNet-50} \\\\'
      '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n& \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{ACC} & \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{ACC} \\\\\n'
      'Dataset & Pub. & Ours & Pub. & Ours & Pub. & Ours & Pub. & Ours \\\\\\midrule\n'
      + '\n'.join(lines) + '\n\\midrule\nAverage & ' + ' & '.join(avg) + ' \\\\\n\\bottomrule\\end{tabular}\n\\end{table}\n')

# resolution: averages over the seven datasets available at every resolution
lines = []
for m in ['glgnet', 'glgnet-local', 'resnet18', 'mobilenetv2', 'vit-tiny', 'swin-t', 'mobilevit-s']:
    if not complete(128, SMALL, m):
        continue
    cs = []
    for z in (28, 64, 128, 224):
        for k in M:
            cs.append(f'{np.mean([mu(z, d, m, k) for d in SMALL]):.3f}' if complete(z, SMALL, m) else '--')
    lines.append(NAME[m].replace('Local only', 'GLGNet-Local') + ' & ' + ' & '.join(cs) + ' \\\\')
write('resolution', '\\begin{table}[t]\\centering\\footnotesize\n'
      '\\caption{Effect of input resolution: test AUC and ACC averaged over the seven datasets used at every resolution '
      '(Derma, Pneumonia, Retina, Breast, Blood, OrganC, OrganS; seeds 0--2).}\\label{tab:resolution}\n'
      '\\begin{tabular}{lcccccccc}\\toprule\n& \\multicolumn{2}{c}{$28^2$} & \\multicolumn{2}{c}{$64^2$} & \\multicolumn{2}{c}{$128^2$} & \\multicolumn{2}{c}{$224^2$} \\\\'
      '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nModel & AUC & ACC & AUC & ACC & AUC & ACC & AUC & ACC \\\\\\midrule\n'
      + '\n'.join(lines) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')

# gate vs local with per-dataset significance (sig_test.py), all available seeds
for z in (28, 128, 224):
    try:
        sig = pd.read_csv(f'results/sig_{z}.csv').set_index('data')
    except FileNotFoundError:
        continue
    ns = int(sig.n_seeds.iloc[0]) if 'n_seeds' in sig else 3
    lines = []
    for d in SMALL:
        s = sig.loc[d]
        lines.append(f"{INFO[d]['python_class']} & {int(s.n_test):,} & "
                     f"{s.auc_a:.3f} & {s.auc_b:.3f} & {s.acc_a:.3f} & {s.acc_b:.3f}"
                     f" & {s.d_auc:+.3f} [{s.auc_lo:+.3f}, {s.auc_hi:+.3f}] & {pfmt(s.p_auc)}"
                     f" & {s.d_acc:+.3f} [{s.acc_lo:+.3f}, {s.acc_hi:+.3f}] & {pfmt(s.p_acc)} \\\\")
    write(f'sig{z}', '\\begin{table*}[t]\\centering\\scriptsize\n'
          f'\\caption{{Gated versus local-only block at ${z}\\times{z}$. $\\Delta$ = GLGNet $-$ GLGNet-Local (mean over {ns} seeds) '
          'with 95\\% paired bootstrap confidence intervals over test images (2{,}000 resamples) and two-sided bootstrap $p$-values. '
          f'The bootstrap captures test-set sampling variability but not seed-to-seed training variability.}}\\label{{tab:sig{z}}}\n'
          '\\begin{tabular}{lrcccccccc}\\toprule\n& & \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{ACC} & & & & \\\\'
          '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nDataset & $n_\\mathrm{test}$ & Gate & Local & Gate & Local & '
          '$\\Delta$AUC [95\\% CI] & $p$ & $\\Delta$ACC [95\\% CI] & $p$ \\\\\\midrule\n'
          + '\n'.join(lines) + '\n\\bottomrule\\end{tabular}\n\\end{table*}\n')

# gate vs local across resolutions: seed means, Wilcoxon over datasets and Bayesian signed-rank (from sig_test.py logs)
import re
full = pd.DataFrame([json.loads(l) for l in open('results/runs.jsonl')])
full = full[(full.tag == 'main') & full.data.isin(SMALL)]
lines = []
for z in (28, 128, 224):
    log = open(f'results/sig_{z}.log', encoding='utf-8').read()
    cs = []
    for k, short in (('test_auc', 'auc'), ('test_acc', 'acc')):
        g, l = (full[(full['size'] == z) & (full.model == m)][k].mean() for m in ('glgnet', 'glgnet-local'))
        w = float(re.search(rf'Wilcoxon over datasets, {short}: p = ([\d.]+)', log).group(1))
        b = re.search(rf'signed-rank {short}: P\(local>gate\)=([\d.]+)\s+P\(equiv[^=]*=([\d.]+)\s+P\(gate>local\)=([\d.]+)', log).groups()
        cs += [f'{g:.3f}', f'{l:.3f}', f'{w:.3f}', f'{float(b[2]):.2f}', f'{float(b[1]):.2f}', f'{float(b[0]):.2f}']
    lines.append(f'${z}^2$ & ' + ' & '.join(cs) + ' \\\\')
write('gate_summary', '\\begin{table*}[t]\\centering\\scriptsize\n'
      '\\caption{Gated block (GLGNet) versus local-only block (GLGNet-Local) at three resolutions, seven datasets, five seeds. '
      'Gate and Local: mean test score. $p_W$: two-sided Wilcoxon signed-rank test over the seven per-dataset means. '
      'P(G), P($\\approx$), P(L): posterior probabilities of the Bayesian signed-rank test that the gated block is practically better, '
      'that the two are practically equivalent (difference below 0.01) and that the local-only block is practically better.}\\label{tab:gate_summary}\n'
      '\\begin{tabular}{lcccccccccccc}\\toprule\n& \\multicolumn{6}{c}{AUC} & \\multicolumn{6}{c}{ACC} \\\\'
      '\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\nInput & Gate & Local & $p_W$ & P(G) & P($\\approx$) & P(L) & Gate & Local & $p_W$ & P(G) & P($\\approx$) & P(L) \\\\\\midrule\n'
      + '\n'.join(lines) + '\n\\bottomrule\\end{tabular}\n\\end{table*}\n')

# efficiency (efficiency.py, measured on an idle GPU)
eff = pd.read_csv('results/efficiency.csv')
lines = []
for m in [m for m in BASE if complete(28, ALLD, m)] + ['glgnet-local']:
    r28, r128 = (eff[(eff.model == m) & (eff['size'] == z)].iloc[0] for z in (28, 128))
    lines.append(NAME[m].replace('Local only', 'GLGNet-Local') + f' & {r28.params_M:.2f}' + ''.join(
        f' & {r.gflops:.3f} & {r.img_per_s:,.0f} & {r.train_ms_per_step:.0f}' for r in (r28, r128)) + ' \\\\')
write('efficiency', '\\begin{table}[t]\\centering\\footnotesize\n'
      '\\caption{Model cost for a 3-channel input and 9 classes, measured on an idle NVIDIA RTX 4000 Ada GPU with bfloat16. '
      'GFLOPs count one multiply--add as one operation. img/s: inference throughput at batch size 256. '
      'ms/step: time of one training step at batch size 128.}\\label{tab:efficiency}\n'
      '\\begin{tabular}{lrrrrrrr}\\toprule\n& & \\multicolumn{3}{c}{$28^2$} & \\multicolumn{3}{c}{$128^2$} \\\\'
      '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nModel & Params (M) & GFLOPs & img/s & ms/step & GFLOPs & img/s & ms/step \\\\\\midrule\n'
      + '\n'.join(lines) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n')
print(eff.round(3).to_string())
