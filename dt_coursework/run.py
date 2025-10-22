import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_wifi_dataset
from src.cv import cross_validate
from src.metrics import save_json
from src.tree import decision_tree_learning
from src.visualize import draw_tree

'''
parameters:
  cm (2D np.ndarray), labels (list/array), title (str), path (str)
functionality:
  Plot a confusion matrix with labels, counts, and colorbar, then save to file.
return:
  None
'''
def plot_confusion(cm, labels, title, path):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')  # show matrix as image
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    # annotate each cell with its count
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # add colorbar
    fig.tight_layout()
    fig.savefig(path, dpi=150)  # write image to disk
    plt.close(fig)              # free figure memory

'''
parameters:
  p (str)
functionality:
  Create directory p if it does not exist.
return:
  None
'''
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

'''
parameters:
  data_path (str), name (str), k (int), seed (int), make_figures (bool)
functionality:
  Load data, run k-fold CV before and after pruning, save metrics and confusion plots,
  dump depth stats, and optionally draw a full tree (clean dataset only).
return:
  None
'''
def run_one(data_path, name, k, seed, make_figures):
    X, y = load_wifi_dataset(data_path)  # load features and labels
    out_dir = os.path.join('outputs', name)
    ensure_dir(out_dir)

    # BEFORE pruning
    res_before = cross_validate(X, y, k=k, seed=seed, prune=False, nested=False)
    save_json(res_before['metrics_before'], os.path.join(out_dir, 'metrics_before.json'))
    cm_before = np.array(res_before['confusion_before'])
    plot_confusion(cm_before, res_before['labels'], f'{name} — Confusion (before)', os.path.join(out_dir, 'cm_before.png'))

    depth_stats = {
        "avg_depth_before": res_before["avg_depth_before"],
        "std_depth_before": res_before["std_depth_before"],
    }

    # AFTER pruning (nested 10-fold)
    res_after = cross_validate(X, y, k=k, seed=seed, prune=True, nested=True, inner_k=10)
    save_json(res_after['metrics_after'], os.path.join(out_dir, 'metrics_after.json'))
    cm_after = np.array(res_after['confusion_after'])
    plot_confusion(cm_after, res_after['labels'], f'{name} — Confusion (after pruning)', os.path.join(out_dir, 'cm_after.png'))

    # accumulate depth stats across before/after
    depth_stats.update({
        "avg_depth_after": res_after["avg_depth_after"],
        "std_depth_after": res_after["std_depth_after"],
        "median_chosen_passes": res_after.get("median_chosen_passes", 0)
    })
    save_json(depth_stats, os.path.join(out_dir, 'depth_before_after.json'))

    # Bonus: visualise full tree on the entire dataset (clean only, by default)
    if make_figures and name == 'clean':
        full_tree, depth = decision_tree_learning(X, y, depth=0)
        draw_tree(full_tree, filename=os.path.join(out_dir, 'tree_full_clean.png'))

'''
parameters:
  None (reads CLI args)
functionality:
  Parse args, choose datasets to run, and invoke run_one per selection.
return:
  None
'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', action='store_true', help='Use wifi_db/clean_dataset.txt')
    ap.add_argument('--noisy', action='store_true', help='Use wifi_db/noisy_dataset.txt')
    ap.add_argument('--data', type=str, default=None, help='Custom data path (TXT)')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--make-figures', action='store_true', help='Also save confusion matrix images and tree figure (clean).')
    args = ap.parse_args()

    to_run = []
    if args.data:
        # use custom path; name from file stem
        to_run.append((args.data, os.path.splitext(os.path.basename(args.data))[0]))
    if args.clean:
        to_run.append(('wifi_db/clean_dataset.txt', 'clean'))
    if args.noisy:
        to_run.append(('wifi_db/noisy_dataset.txt', 'noisy'))
    if not to_run:
        ap.error('Select at least one dataset via --clean/--noisy or provide --data PATH.')

    for path, name in to_run:
        run_one(path, name, args.k, args.seed, args.make_figures)

if __name__ == '__main__':
    main()
