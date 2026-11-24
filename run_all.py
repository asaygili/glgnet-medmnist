"""Full experiment grid. Any number of workers can run in parallel: each picks the next run that is
neither finished (results/runs.jsonl) nor claimed (results/claims/), so the grid can grow while they run.
Usage: python run_all.py            (clear results/claims/ first if workers were killed mid-run)
"""
import ctypes, json, os, subprocess, sys

# Keep Windows from idle-sleeping while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released on exit.
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

ALL = ['pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist',
       'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
SMALL = ['dermamnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'organcmnist', 'organsmnist']
SEEDS = [0, 1, 2]

# (data, model, seed, size), in priority order
grid = [(d, m, s, 28) for m in ('glgnet', 'resnet18', 'resnet50') for d in ALL for s in SEEDS]
grid += [(d, m, s, 28) for m in ('mobilenetv2', 'shufflenetv2', 'efficientnetb0') for d in SMALL for s in SEEDS]
grid += [(d, m, s, 28) for m in ('glgnet-sum', 'glgnet-local', 'glgnet-global') for d in SMALL for s in SEEDS]
grid += [(d, m, s, 28) for m in ('mobilenetv2', 'shufflenetv2', 'efficientnetb0') for d in ALL if d not in SMALL for s in SEEDS]
grid += [(d, m, s, 128) for m in ('glgnet', 'resnet18', 'glgnet-local') for d in SMALL for s in SEEDS]
grid += [(d, m, s, 64) for m in ('glgnet', 'resnet18') for d in SMALL for s in SEEDS]
grid += [(d, 'mobilenetv2', s, z) for z in (128, 64) for d in SMALL for s in SEEDS]
# revision: two more seeds for the key gate-vs-local comparison, then transformer / hybrid baselines
NEW = ('vit-tiny', 'mobilevit-s')  # swin-t dropped: it does not converge under this protocol (AUC 0.69 on the small sets)
grid += [(d, m, s, z) for z in (128, 28) for m in ('glgnet', 'glgnet-local') for d in SMALL for s in (3, 4)]
# 224 px: does the gain of the gated block keep growing with resolution? (five seeds, seed-major order)
grid += [(d, m, s, 224) for s in range(5) for m in ('glgnet', 'glgnet-local', 'resnet18') for d in SMALL]
grid += [(d, m, s, 28) for m in NEW for d in SMALL for s in SEEDS]
grid += [(d, m, s, 128) for m in NEW for d in SMALL for s in SEEDS]
grid += [(d, m, s, 28) for m in NEW for d in ALL if d not in SMALL for s in SEEDS]


def finished():
    if not os.path.exists('results/runs.jsonl'):
        return set()
    rs = [json.loads(l) for l in open('results/runs.jsonl')]
    return {(r['data'], r['model'], r['seed'], r['size']) for r in rs if r['tag'] == 'main'}


os.makedirs('results/claims', exist_ok=True)
fails = 0
while fails < 3:  # three failed runs in a row: stop instead of burning through the grid
    done = finished()
    for d, m, s, z in grid:
        if (d, m, s, z) in done:
            continue
        try:  # atomic claim
            os.close(os.open(f'results/claims/{d}_{m}_s{s}_{z}', os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            continue
    else:
        print('nothing left', flush=True)
        break
    print(f'start {d} {m} seed{s} size{z}', flush=True)
    rc = subprocess.run([sys.executable, '-u', 'train.py', '--data', d, '--model', m, '--seed', str(s),
                         '--size', str(z), '--tag', 'main']).returncode
    fails = fails + 1 if rc else 0
    if rc:  # release the claim so the run is retried later
        print(f'FAILED {d} {m} seed{s} size{z}', flush=True)
        os.remove(f'results/claims/{d}_{m}_s{s}_{z}')
