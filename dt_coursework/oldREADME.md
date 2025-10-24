# Decision Trees from Scratch — WIFI Location Coursework (CO553/70050)

This repo contains a **from-scratch** implementation of decision trees that handle **continuous features** and **multi-class labels**, 10‑fold cross‑validation, **reduced‑error pruning** with **nested 10‑fold CV**, a basic **tree visualizer**, and full metric reporting — using **only NumPy, Matplotlib, and the Python standard library**.

## Project layout
```
dt_coursework/
├─ requirements.txt
├─ README.md
├─ report_template.md
├─ run.py
└─ src/
   ├─ data.py
   ├─ tree.py
   ├─ prune.py
   ├─ metrics.py
   ├─ cv.py
   ├─ visualize.py
   └─ utils.py
```

## Datasets
Place or reference the provided files as:
```
wifi_db/clean_dataset.txt
wifi_db/noisy_dataset.txt
```
(These are the same files you already have in your coursework folder.)

> By default, the scripts expect to be run **from the coursework root folder** that also contains the `wifi_db/` directory. Use `--clean` and `--noisy` to run on either dataset; or pass a custom path with `--data`.

## Quick start

### 0) Create & activate a venv (recommended)
```bash
python -m venv venv
source venv/bin/activate            # on macOS/Linux
# OR
.env\Scripts\activate           # on Windows PowerShell
```

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Run everything (metrics + pruning + figures)
From the directory where `wifi_db/` lives:
```bash
python -m src.data --peek wifi_db/clean_dataset.txt
python run.py --clean --noisy --k 10 --make-figures
```

This will create:
```
outputs/
  clean/
    cm_before.png / cm_after.png
    metrics_before.json / metrics_after.json
    depth_before_after.json
    tree_full_clean.png      # (bonus) visualization on full clean dataset
  noisy/
    cm_before.png / cm_after.png
    metrics_before.json / metrics_after.json
    depth_before_after.json
```

### 3) (Optional) Only one dataset
```bash
python run.py --clean --k 10 --make-figures
# or
python run.py --noisy --k 10
```

### 4) Repro tips
* Set a seed for the K‑Fold splitter (default: `--seed 42`).
* All outputs are deterministic given the seed.

## What the code does

- **Step 1 (Load)** — `src/data.py`: loads the 2000x8 NumPy arrays; last column is the room label (1..4).
- **Step 2 (Learn)** — `src/tree.py`: recursive CART‑style binary splits with information gain (entropy). For each feature, we only consider thresholds between **distinct sorted values**.
- **Bonus (Visualize)** — `src/visualize.py`: draws a simple box‑and‑arrow tree using Matplotlib.
- **Step 3 (Evaluate)** — `src/cv.py` + `src/metrics.py`: 10‑fold CV; aggregate a single 4x4 **confusion matrix**, **accuracy**, **per‑class precision/recall**, and **F1**.
- **Step 4 (Prune)** — `src/prune.py`: **reduced‑error pruning** driven by validation error. For nested 10‑fold CV, the inner CV estimates the **number of pruning passes** to apply on the outer train fold. We refit on the full outer‑train split and prune for the selected number of passes, then evaluate on the outer test split.

## CLI reference

```bash
python run.py [--clean] [--noisy] [--data PATH] [--k 10] [--seed 42] [--make-figures]
```

- `--clean` / `--noisy` : convenience flags that set `--data` to the standard paths.
- `--data PATH`         : custom dataset path (TXT).
- `--k`                 : folds for CV (default 10).
- `--seed`              : RNG seed for fold shuffling (default 42).
- `--make-figures`      : also saves confusion matrices and, for the clean set, a full‑dataset tree figure.

## Report
Use `report_template.md` as a starting point. After running, copy metrics from the JSON files in `outputs/`.

---

**Note**: The implementation uses pure NumPy and Python; no `scikit‑learn` or other ML libs are used.
