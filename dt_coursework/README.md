# Intro to ML: Decision Trees (COMP70050)

From-scratch decision tree implementation for indoor room recognition from WiFi RSSI.

- Continuous features, multi-class labels
- 10-fold CV evaluation and pruning
- Used libraries: NumPy, Matplotlib, Python stdlib
- Target Python: **3.12.x**

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
│     └─ visualize.py
├─ wifi_db/
│  ├─ clean_dataset.txt
│  └─ noisy_dataset.txt
├─ outputs/                # outputs are generated once the pipeline is executed
├─ requirements.txt
└─ README.md
```

**Datasets** should be located at:

```
wifi_db/clean_dataset.txt
wifi_db/noisy_dataset.txt
```

---

## Environment (choose ONE)

### Option A — Use the shared DoC lab virtualenv (lab machines)

```bash
source /vol/lab/ml/intro2ml/bin/activate
# ... run your commands ...
deactivate
```

This environment matches the lab setup and already includes the required packages.

### Option B — Create a local virtualenv (your laptop or custom env)

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: don’t commit `venv/` to git (it’s in `.gitignore`).

---

## Quick Start

Run from the repo root (where `wifi_db/` lives).

**macOS/Linux**

```bash
python dt_coursework/run.py --clean --noisy --k 10 --make-figures
```

**Windows (PowerShell)**

```powershell
python dt_coursework\run.py --clean --noisy --k 10 --make-figures
```

This will:

- Train/evaluate with 10-fold CV on **clean** and **noisy**
- Perform pruning (if enabled in the code)
- Save figures/metrics in `outputs/`

### Only one dataset

```bash
python dt_coursework/run.py --clean --k 10 --make-figures
# or
python dt_coursework/run.py --noisy --k 10

# or this for running the code on a custom dataset
python dt_coursework/run.py --data wifi_db/noisy_dataset.txt --k 10 --make-figures
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
    confusion_before.png
    confusion_after.png
    accuracy.png
    recall.png
    precision.png
    f1.png
    depth_and_leaves_comparison.png
    tree_full_clean.png
    metrics_before.json
    metrics_after.json
  noisy/
    ... same structure ...
```

Use these artefacts to populate the report sections (Confusion matrix, Accuracy, per-class Recall & Precision, F1; and after pruning metrics).

---

## Reproducibility

- Deterministic given `--seed` (default 42).
- Run commands from the repo root so relative paths to `wifi_db/` and `outputs/` resolve.

---

## Troubleshooting

- **Wrong Python / modules missing**: activate the correct environment (Option A or B) and check `python --version`.
- **File not found**: ensure `wifi_db/` exists and you are in the repo root.
- **Windows paths**: use backslashes for the script path (as shown above).
