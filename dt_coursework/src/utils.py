import numpy as np

def unique_labels(y):
    """Return sorted unique labels as int array."""
    return np.unique(y.astype(int))

def majority_label(y):
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])

def label_counts(y, labels=None):
    """Counts of each label in `labels` order."""
    if labels is None:
        labels = unique_labels(y)
    counts = np.zeros(len(labels), dtype=int)
    for i, lab in enumerate(labels):
        counts[i] = np.sum(y == lab)
    return counts

def entropy_from_counts(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def information_gain(parent_counts, left_counts, right_counts):
    n_parent = parent_counts.sum()
    n_left = left_counts.sum()
    n_right = right_counts.sum()
    if n_left == 0 or n_right == 0:
        return 0.0
    H_parent = entropy_from_counts(parent_counts)
    rem = (n_left / n_parent) * entropy_from_counts(left_counts) + (n_right / n_parent) * entropy_from_counts(right_counts)
    return H_parent - rem
