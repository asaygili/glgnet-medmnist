"""Mean gate value (1 = local, 0 = global) per block for each dataset's GLGNet checkpoints.
Usage: python gate_analysis.py [size=28]  -> results/gates_{size}.csv, paper/figures/gates_{size}.pdf"""
import glob, os, sys
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from medmnist import INFO
import models

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 28
sz = '' if SIZE == 28 else f'_{SIZE}'
rows = []
for ck in sorted(glob.glob(f'checkpoints/*{sz}_glgnet_s*.pt')):
    base, seed = os.path.basename(ck).split('_glgnet_s')
    data = base[:-len(sz)] if sz else base
    if data not in INFO:  # e.g. a '_128' checkpoint matched by the 28px glob
        continue
    seed = int(seed[:-3])
    z = np.load(f'data/{data}{sz}.npz')
    x = torch.from_numpy(z['test_images'][:2000])
    x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
    net = models.build('glgnet', x.shape[1], len(INFO[data]['label']), SIZE).cuda().eval()
    net.load_state_dict(torch.load(ck, map_location='cuda'))
    blocks = [m for m in net.modules() if isinstance(m, models.GLGBlock)]
    sums = np.zeros(len(blocks))
    with torch.no_grad():
        for i in range(0, len(x), 250):
            xb = (x[i:i + 250].cuda().float() / 255 - 0.5) / 0.5
            net(xb)
            sums += [b.last_gate.mean().item() * len(xb) for b in blocks]
    for b, s in enumerate(sums / len(x)):
        rows.append(dict(data=data, seed=seed, block=b + 1, gate=s))

df = pd.DataFrame(rows)
df.to_csv(f'results/gates_{SIZE}.csv', index=False)
m = df.groupby(['data', 'block']).gate.mean().unstack()
m.index = [INFO[d]['python_class'] for d in m.index]
m = m.loc[m.mean(axis=1).sort_values(ascending=False).index]
print(m.round(3).to_string())
print('stage means:', {s: round(float(m[c].values.mean()), 3) for s, c in (('S1', [1, 2]), ('S2', [3, 4]), ('S3', [5, 6, 7, 8]))})

os.makedirs('paper/figures', exist_ok=True)
fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(m) + 1))
im = ax.imshow(m.values, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(m.shape[1]), [f'S{1 if b <= 2 else 2 if b <= 4 else 3}B{b}' for b in m.columns])
ax.set_yticks(range(len(m)), m.index)
for (i, j), v in np.ndenumerate(m.values):
    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
fig.colorbar(im, ax=ax, label='mean gate  (1 = local, 0 = global)')
ax.set_xlabel('block (stage, index)')
ax.set_title(f'${SIZE}\\times{SIZE}$ input', fontsize=9)
fig.tight_layout()
fig.savefig(f'paper/figures/gates_{SIZE}.pdf')
