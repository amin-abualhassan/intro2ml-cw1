from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any
from .utils import unique_labels, label_counts, information_gain

Node = Dict[str, Any]


def _best_split_for_feature(x, y, labels):
    '''
    parameters:
    x (1D np.ndarray), y (1D array-like), labels (array-like)

    functionality:
    For one feature, scan valid split points and pick the threshold with max information gain.
    
    return:
    (best_gain: float, best_threshold: float|None)
    '''
    n = len(y)
    order = np.argsort(x, kind="mergesort")  # stable sort preserves order on ties
    x_sorted = x[order]
    y_sorted = y[order]

    parent_counts = label_counts(y_sorted, labels)

    # Precompute cumulative counts per label along sorted order
    K = len(labels)
    cum = np.zeros((n, K), dtype=int)
    for i, lab in enumerate(labels):
        cum[:, i] = (y_sorted == lab).astype(int).cumsum()

    best_gain = -1.0
    best_thr = None

    # Only consider cuts between distinct feature values
    distinct = np.where(np.diff(x_sorted) != 0)[0]  # valid split indices (between i and i+1)
    if distinct.size == 0:
        return 0.0, None

    for i in distinct:
        left_counts = cum[i, :]                       # counts up to i (inclusive)
        right_counts = parent_counts - left_counts    # remaining counts
        gain = information_gain(parent_counts, left_counts, right_counts)
        if gain > best_gain:
            best_gain = gain
            # threshold midway between the two neighboring values
            best_thr = (x_sorted[i] + x_sorted[i+1]) * 0.5

    return best_gain if best_gain > 0 else 0.0, best_thr


def find_split(X, y, labels):
    '''
    parameters:
    X (2D np.ndarray), y (1D array-like), labels (array-like)

    functionality:
    Evaluate all features with the single-feature splitter; keep the best (attr, threshold, gain).
    
    return:
    (best_attr: int|None, best_threshold: float|None, best_gain: float)
    '''
    n_features = X.shape[1]
    best = (None, None, 0.0)  # (attr, thr, gain)
    for j in range(n_features):
        gain, thr = _best_split_for_feature(X[:, j], y, labels)
        if gain > best[2]:
            best = (j, thr, gain)
    return best


def make_leaf(y, depth):
    '''
    parameters:
    y (1D array-like), depth (int)

    functionality:
    Build a leaf node: predicted label, depth, per-class counts, and label list.
    
    return:
    Node (dict)
    '''
    
    # Correct prediction using the actual labels present in y
    labs = unique_labels(y)
    pred = int(labs[np.argmax(label_counts(y, labs))])

    return {
        "leaf": True,
        "prediction": pred,
        "depth": depth,
        "class_counts": label_counts(y, labels=labs),
        "labels": labs.tolist(),
    }


def decision_tree_learning(X, y, depth=0, num_of_leaf=0) -> Tuple[Node, int, int]:
    '''
    parameters:
    X (2D np.ndarray), y (1D array-like), depth (int, default 0)

    functionality:
    Recursively grow a decision tree using info gain. Stop on pure nodes or no useful split.
    
    return:
    (node: Node, max_depth: int)
    '''
    labs = unique_labels(y)

    # Stop if pure (single label)
    if len(labs) == 1:
        leaf = make_leaf(y, depth)
        num_of_leaf += 1
        return leaf, depth, num_of_leaf

    # Find best split across all features
    attr, thr, gain = find_split(X, y, labs)

    # Stop if no split or no gain
    if attr is None or thr is None or gain <= 0.0:
        leaf = make_leaf(y, depth)
        num_of_leaf += 1
        return leaf, depth, num_of_leaf

    # Partition data by the chosen threshold
    left_mask = X[:, attr] <= thr
    right_mask = ~left_mask

    # Guard against degenerate split
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        leaf = make_leaf(y, depth)
        num_of_leaf += 1
        return leaf, depth, num_of_leaf

    # Recurse on children
    left_node, l_depth, l_num_of_leaf= decision_tree_learning(X[left_mask], y[left_mask], depth+1, num_of_leaf)
    right_node, r_depth, r_num_of_leaf= decision_tree_learning(X[right_mask], y[right_mask], depth+1,num_of_leaf)

    # Build internal node
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
    return node, max(l_depth, r_depth), l_num_of_leaf+r_num_of_leaf


def predict_one(node: Node, x):
    '''
    parameters:
    node (Node), x (1D np.ndarray)

    functionality:
    Traverse the tree from root to leaf using thresholds; return the leaf prediction.
    
    return:
    int (predicted label)
    '''
    while not node.get("leaf", False):
        # Go left if value <= threshold; otherwise right
        if x[node["attr"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return int(node["prediction"])


def predict(node: Node, X: np.ndarray):
    '''
    parameters:
    node (Node), X (2D np.ndarray)

    functionality:
    Predict a label for each row of X using predict_one.
    
    return:
    np.ndarray (dtype=int) of predictions
    '''
    return np.array([predict_one(node, row) for row in X], dtype=int)


def tree_size(node: Node):
    '''
    parameters:
    node (Node)

    functionality:
    Count total number of nodes in the tree (leaves + internals).
    
    return:
    int
    '''
    if node.get("leaf", False):
        return 1
    return 1 + tree_size(node["left"]) + tree_size(node["right"])


def tree_max_depth(node: Node):
    '''
    parameters:
    node (Node)

    functionality:
    Compute the maximum depth recorded in the tree.

    return:
    int
    '''
    if node.get("leaf", False):
        return node.get("depth", 0)
    return max(tree_max_depth(node["left"]), tree_max_depth(node["right"]))

def tree_count_leaves(node: Node, num_of_leaves=0):
    '''
    parameters:
    node (Node)

    functionality:
    Compute the number of leaves recorded in the tree.

    return:
    int
    '''
    if node.get("leaf", True):
        num_of_leaves += 1
        return num_of_leaves
    return tree_count_leaves(node["left"],num_of_leaves) + tree_count_leaves(node["right"],num_of_leaves)
