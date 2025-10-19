from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any
from .utils import unique_labels, label_counts, information_gain

Node = Dict[str, Any]

def _best_split_for_feature(x, y, labels):
    """Return (gain, threshold) for a single feature vector x and labels y."""
    n = len(y)
    order = np.argsort(x, kind="mergesort")  # stable
    x_sorted = x[order]
    y_sorted = y[order]
    parent_counts = label_counts(y_sorted, labels)
    # cumulative counts for each label along the sorted order
    K = len(labels)
    cum = np.zeros((n, K), dtype=int)
    for i, lab in enumerate(labels):
        cum[:, i] = (y_sorted == lab).astype(int).cumsum()
    best_gain = -1.0
    best_thr = None
    # consider thresholds between distinct values
    distinct = np.where(np.diff(x_sorted) != 0)[0]  # indices where a split can occur between i and i+1
    if distinct.size == 0:
        return 0.0, None
    for i in distinct:
        left_counts = cum[i, :]
        right_counts = parent_counts - left_counts
        gain = information_gain(parent_counts, left_counts, right_counts)
        if gain > best_gain:
            best_gain = gain
            best_thr = (x_sorted[i] + x_sorted[i+1]) * 0.5
    return best_gain if best_gain > 0 else 0.0, best_thr

def find_split(X, y, labels):
    """Search over all features; return (best_attr, best_threshold, best_gain)."""
    n_features = X.shape[1]
    best = (None, None, 0.0)
    for j in range(n_features):
        gain, thr = _best_split_for_feature(X[:, j], y, labels)
        if gain > best[2]:
            best = (j, thr, gain)
    return best

def make_leaf(y, depth):
    counts = label_counts(y)
    pred = int(np.argmax(counts) + 1)  # labels are 1..K when counts computed w/o labels; fix below
    # Correct pred using actual labels
    labs = unique_labels(y)
    pred = int(labs[np.argmax(label_counts(y, labs))])
    return {
        "leaf": True,
        "prediction": pred,
        "depth": depth,
        "class_counts": label_counts(y, labels=labs),
        "labels": labs.tolist(),
    }

def decision_tree_learning(X, y, depth=0) -> Tuple[Node, int]:
    """Recursive decision tree learner for continuous features & multi-class labels.
    Returns (node, max_depth).
    """
    labs = unique_labels(y)
    if len(labs) == 1:
        leaf = make_leaf(y, depth)
        return leaf, depth
    attr, thr, gain = find_split(X, y, labs)
    if attr is None or thr is None or gain <= 0.0:
        leaf = make_leaf(y, depth)
        return leaf, depth
    # split dataset
    left_mask = X[:, attr] <= thr
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        leaf = make_leaf(y, depth)
        return leaf, depth
    left_node, l_depth = decision_tree_learning(X[left_mask], y[left_mask], depth+1)
    right_node, r_depth = decision_tree_learning(X[right_mask], y[right_mask], depth+1)
    node = {
        "leaf": False,
        "attr": int(attr),
        "threshold": float(thr),
        "left": left_node,
        "right": right_node,
        "depth": depth,
        "class_counts": label_counts(y, labs),
        "labels": labs.tolist(),
        "gain": float(gain),
    }
    return node, max(l_depth, r_depth)

def predict_one(node: Node, x):
    while not node.get("leaf", False):
        if x[node["attr"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return int(node["prediction"])

def predict(node: Node, X: np.ndarray):
    return np.array([predict_one(node, row) for row in X], dtype=int)

def tree_size(node: Node):
    if node.get("leaf", False):
        return 1
    return 1 + tree_size(node["left"]) + tree_size(node["right"])

def tree_max_depth(node: Node):
    if node.get("leaf", False):
        return node.get("depth", 0)
    return max(tree_max_depth(node["left"]), tree_max_depth(node["right"]))
