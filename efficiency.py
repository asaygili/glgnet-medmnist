"""Params, FLOPs (1 image, 3 channels, 9 classes), inference throughput and training-step time -> results/efficiency.csv
Run on an idle GPU: timings taken while other jobs train are not comparable."""
import time
import pandas as pd, torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
import models

MODELS = ['resnet18', 'resnet50', 'mobilenetv2', 'shufflenetv2', 'efficientnetb0', 'vit-tiny', 'swin-t', 'mobilevit-s',
          'glgnet', 'glgnet-local']


def timed(fn, warm, n):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


rows = []
for size in (28, 128):
    for m in MODELS:
        net = models.build(m, 3, 9, size).cuda().eval().to(memory_format=torch.channels_last)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            net(torch.randn(1, 3, size, size, device='cuda'))
        x = torch.randn(256, 3, size, size, device='cuda').contiguous(memory_format=torch.channels_last)

        def infer():
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16): net(x)
        t_inf = min(timed(infer, 10, 30) for _ in range(3))  # best of three repeats

        net.train()
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        xb, yb = x[:128], torch.randint(0, 9, (128,), device='cuda')

        def step():  # one training step exactly as in train.py (batch 128, bf16, Adam)
            with torch.autocast('cuda', torch.bfloat16): loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        t_step = min(timed(step, 5, 20) for _ in range(3))
        rows.append(dict(model=m, size=size, params_M=sum(p.numel() for p in net.parameters()) / 1e6,
                         gflops=fc.get_total_flops() / 2e9,  # FlopCounterMode counts 2 per multiply-add; we report multiply-adds
                         img_per_s=256 / t_inf, train_ms_per_step=1000 * t_step))
        print(rows[-1], flush=True)
        del net, opt
pd.DataFrame(rows).to_csv('results/efficiency.csv', index=False)
