"""Generate LaTeX tables in paper/tables/ from MedMNIST INFO and results/runs.jsonl."""
import os
from medmnist import INFO

os.makedirs('paper/tables', exist_ok=True)
MOD = {'pathmnist': 'Colon pathology', 'chestmnist': 'Chest X-ray', 'dermamnist': 'Dermatoscope',
       'octmnist': 'Retinal OCT', 'pneumoniamnist': 'Chest X-ray', 'retinamnist': 'Fundus camera',
       'breastmnist': 'Breast ultrasound', 'bloodmnist': 'Blood cell microscope',
       'tissuemnist': 'Kidney cortex microscope', 'organamnist': 'Abdominal CT (axial)',
       'organcmnist': 'Abdominal CT (coronal)', 'organsmnist': 'Abdominal CT (sagittal)'}
TASK = {'multi-label, binary-class': 'Multi-label', 'binary-class': 'Binary',
        'multi-class': 'Multi-class', 'ordinal-regression': 'Ordinal'}

rows = []
for d, mod in MOD.items():
    i, n = INFO[d], INFO[d]['n_samples']
    rows.append(f"{i['python_class']} & {mod} & {TASK[i['task']]} ({len(i['label'])}) & "
                f"{n['train']:,} & {n['val']:,} & {n['test']:,} \\\\")

with open('paper/tables/datasets.tex', 'w', encoding='utf-8') as f:
    f.write('\\begin{table}[t]\\centering\\small\n'
            '\\caption{The twelve 2D datasets of MedMNIST v2 used in this study.}\\label{tab:datasets}\n'
            '\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{lllrrr}\\toprule\n'
            'Dataset & Modality & Task (\\#classes) & Train & Val & Test \\\\\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\\end{adjustbox}\n\\end{table}\n')
print(open('paper/tables/datasets.tex', encoding='utf-8').read())

