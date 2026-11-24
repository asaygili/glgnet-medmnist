# Does global attention help lightweight medical image classifiers?

Code and complete run logs for the paper *"Does global attention help lightweight medical image classifiers? A controlled multi-resolution study on MedMNIST"* (A. Saygılı, submitted to Biomedical Signal Processing and Control).

GLGNet is a 4.4M-parameter network whose blocks fuse a multi-scale depthwise-convolution branch and a pooled-key attention branch through a channel gate. GLGNet-Local removes the attention branch and the gate and serves as the matched convolution-only control.

## Contents

| File | Purpose |
|---|---|
| `models.py` | GLGNet, its ablation variants (`glgnet-sum`, `glgnet-local`, `glgnet-global`) and all baselines |
| `train.py` | Trains and evaluates one model on one dataset with the official MedMNIST v2 recipe; appends a line to `results/runs.jsonl` |
| `run_all.py` | Full experiment grid (701 runs); several workers can run in parallel |
| `sig_test.py` | Paired bootstrap, McNemar and Bayesian signed-rank tests, gate vs local |
| `gate_analysis.py` | Learned gate values per block and dataset |
| `efficiency.py` | Parameters, FLOPs, throughput and training-step time |
| `summarize.py`, `make_tables.py`, `make_results_tables.py` | Result summaries and the LaTeX tables of the paper |
| `results/` | Logs of every run (`runs.jsonl`) and the outputs of the analysis scripts |

## Setup

Python 3.11, PyTorch 2.x with CUDA. Install the dependencies:

```
pip install -r requirements.txt
```

Download the MedMNIST v2 / MedMNIST+ `.npz` files into `data/` (e.g. `data/pathmnist.npz`, `data/pathmnist_128.npz`):

```python
from medmnist import INFO
import medmnist
for name in INFO:
    if not name.endswith('3d'):
        for size in (28, 64, 128, 224):
            getattr(medmnist, INFO[name]['python_class'])(split='test', download=True, size=size, root='data')
```

## Usage

```
python train.py --data breastmnist --model glgnet --size 28 --seed 0
python run_all.py                 # the full grid
python sig_test.py 128            # significance tests at 128x128
python gate_analysis.py 28
python make_results_tables.py
```

Training protocol: Adam (lr 1e-3), batch size 128, 100 epochs, learning rate x0.1 at epochs 50 and 75, no augmentation, checkpoint with the best validation AUC, bf16 autocast.

## License

MIT
