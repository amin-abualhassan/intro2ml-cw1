# Intro to ML: Decision Trees (COMP70050)

From-scratch decision tree implementation for indoor room recognition from WiFi RSSI.

- Continuous features, multi-class labels
- 10-fold CV evaluation and pruning
- Used libraries: NumPy, Matplotlib, Python stdlib
- Target Python: 3.12.x

---

## Project Layout

```
intro2ml-cw1/
├─ dt_coursework/
│  ├─ run.py
│  └─ src/
│     ├─ cv.py
│     ├─ data.py
│     ├─ metrics.py
│     ├─ prune.py
│     ├─ tree.py
│     ├─ utils.py
│     ├─ visualize.py
│     └─ analytics.ipynb  # python notebook to analyze the decision tree result (for the report)
├─ wifi_db/
│  ├─ clean_dataset.txt
│  └─ noisy_dataset.txt
├─ outputs/                # outputs are generated once the pipeline is executed
├─ requirements.txt
└─ README.md
```

Datasets should be located at:

```
wifi_db/clean_dataset.txt
wifi_db/noisy_dataset.txt
```

---

## Environment (choose ONE)

### Option A — Use the shared DoC lab virtualenv (lab machines)

```bash
source /vol/lab/ml/intro2ml/bin/activate
# ... run code commands shown in the Quick Start section ...
deactivate
```
### Option B — Create a local virtualenv (your laptop or custom env)

macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# ... run code commands shown in the Quick Start section ...

deactivate
```


---

## Quick Start

Run from the repo root (where `wifi_db/` lives).

macOS/Linux

```bash
python dt_coursework/run.py --clean --noisy --k 10 --make-figures
```

This will:

- Train/evaluate with 10-fold CV on clean and noisy
- Perform pruning (if enabled in the code)
- Save figures/metrics in `outputs/`

### Only one dataset

#### Clean Dataset
```bash
python dt_coursework/run.py --clean --k 10 --make-figures
```

#### Noisy Dataset
```bash
python dt_coursework/run.py --noisy --k 10
```

#### Custom Dataset
# Running the code on a custom dataset located at custom_dataset_file_path
```bash
python dt_coursework/run.py --data custom_dataset_file_path --k 10 --make-figures
```

### Useful CLI flags

```
--clean / --noisy      # convenience dataset selectors
--data PATH            # custom dataset .txt (overrides the above)
--k 10                 # number of CV folds (default: 10)
--seed 42              # RNG seed (default: 42)
--make-figures         # write plots to outputs/
```

---

## Outputs

```
outputs/
  clean/
    cm_after.png
    depth_before_after.json
    metrics_before.json
    cm_before.png
    metrics_after.json
    tree_full_clean.png # you need to zoom to see the tree details

  noisy/
    cm_after.png
    depth_before_after.json
    metrics_before.json
    cm_before.png
    metrics_after.json
    tree_full_noisy.png # you need to zoom a lot to see the tree details

  ... same structure if code is ran for a custom dataset ...
```

---

## Reproducibility

- Deterministic given `--seed` (default 42).
- Run commands from the repo root so relative paths to `wifi_db/` and `outputs/` resolve.

---

## Troubleshooting

- Wrong Python / modules missing: activate the correct environment (Option A or B) and check `python --version`.
- File not found: ensure `wifi_db/` exists and you are in the repo root.