from __future__ import annotations
import copy
import numpy as np
from typing import Tuple, Dict, Any, List
from .tree import predict, tree_max_depth
from .utils import majority_label

Node = Dict[str, Any]

def _iter_prunable_nodes(node: Node, parent=None, side=None):
    """Yield (node, parent, side) for nodes whose children are both leaves."""
    if node.get("leaf", False):
        return
    left = node["left"]
    right = node["right"]
    if left.get("leaf", False) and right.get("leaf", False):
        yield (node, parent, side)
    # Recurse
    yield from _iter_prunable_nodes(left, node, "left")
    yield from _iter_prunable_nodes(right, node, "right")

def validation_error(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> float:
    if X_val.size == 0:
        return 0.0
    preds = predict(node, X_val)
    return float(np.mean(preds != y_val))

def _prune_node_in_place(node: Node):
    """Replace `node` by a single majority-class leaf using the node's class_counts."""
    # Majority label from training counts stored at node
    if "labels" in node and "class_counts" in node:
        labs = np.array(node["labels"], dtype=int)
        counts = np.array(node["class_counts"], dtype=int)
        pred = int(labs[np.argmax(counts)])
    else:
        pred = node.get("prediction", None)
        if pred is None:
            # Fallback: if missing, descend to collect leaves' counts
            left_pred = node["left"].get("prediction", None)
            right_pred = node["right"].get("prediction", None)
            pred = left_pred if left_pred is not None else right_pred
        pred = int(pred)
    node.clear()
    node.update({
        "leaf": True,
        "prediction": pred,
        "depth": node.get("depth", 0),
    })

def prune_one_pass(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> int:
    """Greedy pass over all prunable nodes. If pruning a node does not increase
    validation error (i.e., err_new <= err_old), prune it. Returns number of nodes pruned.
    """
    pruned = 0
    # We will scan repeatedly in case the structure changes as we prune
    changed = True
    while changed:
        changed = False
        for cand, parent, side in list(_iter_prunable_nodes(node)):
            base_err = validation_error(node, X_val, y_val)
            # try prune (on a temp copy of the candidate only by simulating)
            snapshot = copy.deepcopy(cand.copy())
            # prune candidate in-place
            _prune_node_in_place(cand)
            new_err = validation_error(node, X_val, y_val)
            if new_err <= base_err + 1e-12:
                pruned += 1
                changed = True
                # keep pruned
            else:
                # revert candidate
                cand.clear()
                cand.update(snapshot)
    return pruned

def pruning_path(node: Node, X_val: np.ndarray, y_val: np.ndarray, pass_limit: int | None = None):
    """Return a list of (tree_copy, val_error) after 0,1,2,... passes until convergence or pass_limit."""
    models: List[Node] = [copy.deepcopy(node)]
    errors: List[float] = [validation_error(models[-1], X_val, y_val)]
    passes = 0
    while True:
        if pass_limit is not None and passes >= pass_limit:
            break
        work = copy.deepcopy(models[-1])
        num_pruned = prune_one_pass(work, X_val, y_val)
        cur_err = validation_error(work, X_val, y_val)
        if num_pruned == 0:  # converged
            models.append(work)
            errors.append(cur_err)
            break
        models.append(work)
        errors.append(cur_err)
        passes += 1
    return models, errors

def prune_with_passes(node: Node, X_val: np.ndarray, y_val: np.ndarray, passes: int):
    """Apply exactly `passes` pruning passes (or until convergence if earlier)."""
    work = copy.deepcopy(node)
    for i in range(passes):
        n = prune_one_pass(work, X_val, y_val)
        if n == 0:
            break
    return work

def count_passes_to_converge(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> int:
    models, _ = pruning_path(node, X_val, y_val)
    # number of passes is len(models)-1 (including unpruned as step 0)
    return max(0, len(models) - 1)

def max_depth_after_prune(node: Node):
    return tree_max_depth(node)
