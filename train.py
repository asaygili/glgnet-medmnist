"""Train/evaluate one model on one MedMNIST 2D dataset with the official MedMNIST v2 recipe.

Usage: python train.py --data pathmnist --model resnet18 --seed 0
Appends one JSON line per run to results/runs.jsonl.
"""
import argparse, json, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from medmnist import INFO
from medmnist.evaluator import getAUC, getACC
import models

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--model', default='resnet18')
p.add_argument('--size', type=int, default=28)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--bs', type=int, default=128)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--tag', default='')
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda'
info = INFO[a.data]
task = info['task']
n_cls = len(info['label'])
multilabel = task == 'multi-label, binary-class'

suffix = '' if a.size == 28 else f'_{a.size}'
z = np.load(f'data/{a.data}{suffix}.npz')

def to_gpu(x, y):  # kept as uint8 on the GPU (128px sets are large); normalised per batch in prep()
    x = torch.from_numpy(x)
    x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
    return x.to(dev).contiguous(memory_format=torch.channels_last), torch.from_numpy(y).to(dev)


def prep(x):
    return (x.float() / 255 - 0.5) / 0.5  # official normalisation

tr, va, te = (to_gpu(z[f'{s}_images'], z[f'{s}_labels']) for s in ('train', 'val', 'test'))
in_ch = tr[0].shape[1]

net = models.build(a.model, in_ch, n_cls, a.size).to(dev).to(memory_format=torch.channels_last)
n_params = sum(p.numel() for p in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * 0.5), int(a.epochs * 0.75)], 0.1)

def loss_fn(out, y):
    return F.binary_cross_entropy_with_logits(out, y.float()) if multilabel else F.cross_entropy(out, y.view(-1).long())

@torch.no_grad()
def evaluate(x, y):
    net.eval()
    outs = []
    eb = 1024 if a.size == 28 else 256  # eval batch; smaller at high resolution to bound memory
    for i in range(0, len(x), eb):
        with torch.autocast('cuda', torch.bfloat16):
            o = net(prep(x[i:i + eb]))
        outs.append(o.float())
    o = torch.cat(outs)
    s = torch.sigmoid(o) if multilabel else o.softmax(1)
    s, yy = s.cpu().numpy(), y.cpu().numpy()
    return getAUC(yy, s, task), getACC(yy, s, task)

best = (-1, None, None)
t0 = time.time()
for ep in range(a.epochs):
    net.train()
    perm = torch.randperm(len(tr[0]), device=dev)
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i + a.bs]
        with torch.autocast('cuda', torch.bfloat16):
            loss = loss_fn(net(prep(tr[0][idx])), tr[1][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sched.step()
    v_auc, v_acc = evaluate(*va)
    if v_auc > best[0]:
        best = (v_auc, ep, evaluate(*te))
        if a.model.startswith('glgnet') and a.tag == 'main':  # kept for gate analysis
            os.makedirs('checkpoints', exist_ok=True)
            sz = '' if a.size == 28 else f'_{a.size}'
            torch.save(net.state_dict(), f'checkpoints/{a.data}{sz}_{a.model}_s{a.seed}.pt')
train_time = time.time() - t0

res = dict(data=a.data, model=a.model, size=a.size, seed=a.seed, tag=a.tag, params=n_params,
           best_epoch=best[1], val_auc=best[0], test_auc=best[2][0], test_acc=best[2][1],
           train_time_s=round(train_time, 1))
print(json.dumps(res))
os.makedirs('results', exist_ok=True)
with open('results/runs.jsonl', 'a') as f:
    f.write(json.dumps(res) + '\n')
