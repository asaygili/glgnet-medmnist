"""Per-dataset significance of GLGNet (gate) vs GLGNet-Local from saved checkpoints (no retraining).

- paired bootstrap over test images of the seed-averaged AUC/ACC difference (95% CI, two-sided p)
- exact McNemar test on accuracy, per seed
- Bayesian signed-rank test over datasets (Benavoli et al., 2017), ROPE = 0.01
Usage: python sig_test.py [size=128]   -> results/sig_{size}.csv
"""
import sys, json, os
import numpy as np, torch, pandas as pd
from scipy.stats import binomtest, ttest_ind, wilcoxon
from medmnist import INFO
from medmnist.evaluator import getAUC, getACC
import models

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
SMALL = ['dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'organcmnist', 'organsmnist']
A, B, NB = 'glgnet', 'glgnet-local', 2000
rng = np.random.default_rng(0)
runs = [json.loads(l) for l in open('results/runs.jsonl')]
sz = '' if SIZE == 28 else f'_{SIZE}'
# every seed for which both models have a checkpoint on every dataset
SEEDS = [s for s in range(10) if all(os.path.exists(f'checkpoints/{d}{sz}_{m}_s{s}.pt') for d in SMALL for m in (A, B))]
print('seeds', SEEDS)


@torch.no_grad()
def predict(d, m, s, x):
    info = INFO[d]
    net = models.build(m, x.shape[1], len(info['label']), SIZE).cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(f'checkpoints/{d}{sz}_{m}_s{s}.pt'))
    out = []
    for i in range(0, len(x), 512):
        with torch.autocast('cuda', torch.bfloat16):
            out.append(net((x[i:i + 512].float() / 255 - 0.5) / 0.5).float())
    o = torch.cat(out)
    return (torch.sigmoid(o) if info['task'].startswith('multi-label') else o.softmax(1)).cpu().numpy()


def correct(y, p, task):  # per-image correctness, consistent with getACC
    if task.startswith('multi-label'):
        return ((p > 0.5) == y).mean(1)
    if task == 'binary-class':
        return (p[:, -1] > 0.5).astype(int) == y[:, 0]
    return p.argmax(1) == y[:, 0]


def bayes_signed_rank(z, rope=0.01, n=20000):
    z = np.concatenate([[0.0], z])  # prior pseudo-observation at 0 with strength 0.5, as in Benavoli et al.
    w = rng.dirichlet(np.r_[0.5, np.ones(len(z) - 1)], n)
    s = z[:, None] + z[None, :]
    probs = [np.einsum('ni,nj,ij->n', w, w, mask.astype(float)) for mask in (s < -2 * rope, abs(s) <= 2 * rope, s > 2 * rope)]
    win = np.argmax(np.stack(probs), 0)
    return [(win == k).mean() for k in range(3)]  # P(local better), P(equivalent), P(gate better)


rows = []
for d in SMALL:
    task = INFO[d]['task']
    z = np.load(f'data/{d}{sz}.npz')
    x = torch.from_numpy(z['test_images'])
    x = (x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)).cuda().contiguous(memory_format=torch.channels_last)
    y = z['test_labels']
    P = {(m, s): predict(d, m, s, x) for m in (A, B) for s in SEEDS}
    for (m, s), p in P.items():  # sanity: must reproduce the logged test AUC
        logged = next(r['test_auc'] for r in runs if (r['data'], r['model'], r['seed'], r['size'], r['tag']) == (d, m, s, SIZE, 'main'))
        assert abs(getAUC(y, p, task) - logged) < 2e-3, (d, m, s, getAUC(y, p, task), logged)

    def diff(idx):
        auc = np.mean([getAUC(y[idx], P[A, s][idx], task) - getAUC(y[idx], P[B, s][idx], task) for s in SEEDS])
        acc = np.mean([getACC(y[idx], P[A, s][idx], task) - getACC(y[idx], P[B, s][idx], task) for s in SEEDS])
        return auc, acc

    obs = diff(np.arange(len(y)))
    boot = np.array([diff(rng.integers(0, len(y), len(y))) for _ in range(NB)])
    mcn = []
    for s in SEEDS:
        ca, cb = correct(y, P[A, s], task), correct(y, P[B, s], task)
        b01, b10 = int(((ca > cb)).sum()), int(((cb > ca)).sum())
        mcn.append(binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0)
    row = dict(data=d, n_test=len(y), n_seeds=len(SEEDS), d_auc=obs[0], d_acc=obs[1])
    for name, fn in (('auc', getAUC), ('acc', getACC)):  # seed means, and Welch t-test across training seeds
        va, vb = ([fn(y, P[m, s], task) for s in SEEDS] for m in (A, B))
        row.update({f'{name}_a': np.mean(va), f'{name}_b': np.mean(vb), f'p_seed_{name}': ttest_ind(va, vb, equal_var=False).pvalue})
    for k, name in enumerate(('auc', 'acc')):
        lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
        p = min(1.0, 2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean()))
        row.update({f'{name}_lo': lo, f'{name}_hi': hi, f'p_{name}': p})
    row.update(mcnemar_s0=mcn[0], mcnemar_s1=mcn[1], mcnemar_s2=mcn[2])
    rows.append(row)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'results/sig_{SIZE}.csv', index=False)
for name in ('auc', 'acc'):
    pl, pe, pg = bayes_signed_rank(df[f'd_{name}'].values)
    print(f'Wilcoxon over datasets, {name}: p = {wilcoxon(df[f"d_{name}"]).pvalue:.3f}')
    print(f'Bayesian signed-rank {name}: P(local>gate)={pl:.3f}  P(equiv, |d|<1pt)={pe:.3f}  P(gate>local)={pg:.3f}')
