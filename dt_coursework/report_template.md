# Short Report — Decision Trees from Scratch (CO553/70050)

## (Bonus) Tree visualisation (clean dataset)
Add `outputs/clean/tree_full_clean.png` here if you generated it.

---

## Step 3 — Evaluation (before pruning)

### Confusion matrix (average over 10 folds)
Insert `outputs/clean/cm_before.png` and `outputs/noisy/cm_before.png`.

### Overall accuracy
- Clean:    (from `outputs/clean/metrics_before.json`)
- Noisy:    (from `outputs/noisy/metrics_before.json`)

### Per‑class precision / recall / F1
Copy the tables from the JSON files (clean & noisy).

### Result analysis (≤ 5 lines)
- Which rooms are correctly recognized? Where are the confusions? Briefly comment.

### Dataset differences (≤ 5 lines)
- Do clean vs noisy differ? Why?

---

## Step 4 — Pruning (and evaluation again)

### Confusion matrix (after pruning; nested 10‑fold)
Insert `outputs/clean/cm_after.png` and `outputs/noisy/cm_after.png`.

### Overall accuracy (after pruning)
- Clean:    (from `outputs/clean/metrics_after.json`)
- Noisy:    (from `outputs/noisy/metrics_after.json`)

### Per‑class precision / recall / F1 (after pruning)
Copy from the JSON files.

### Result analysis after pruning (≤ 5 lines)
- How did pruning change performance?

### Depth analysis (≤ 5 lines)
- Compare average tree depths before vs after pruning (see `depth_before_after.json` in each dataset folder).
- Comment on the relationship between maximal depth and prediction accuracy.
