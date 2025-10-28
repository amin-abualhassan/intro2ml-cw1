from __future__ import annotations
import copy
import numpy as np
from typing import Tuple, Dict, Any, List
from src.tree import predict, tree_max_depth

Node = Dict[str, Any]

def _iter_prunable_nodes(node: Node, parent=None, side=None):
    '''
    parameters:
        node (Node)
        parent (Node or None)
        side (str or None)
    functionality:
        Recursively finds nodes whose left and right children are both leaves.
    return:
        Yields tuples of (node, parent, side) for prunable nodes.
    '''
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
    '''
    parameters:
        node (Node)
        X_val (np.ndarray)
        y_val (np.ndarray)
    functionality:
        Computes the validation error by comparing predictions to true labels.
    return:
        float validation error rate.
    '''
    if X_val.size == 0:
        return 0.0
    preds = predict(node, X_val)
    return float(np.mean(preds != y_val))

def _prune_node_in_place(node: Node):
    '''
    parameters:
        node (Node)
    functionality:
        Converts the node into a leaf predicting the majority class of samples.
        Removes its left and right children.
    return:
        None
    '''
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
    depth = node.get("depth", 0)
    node.clear()
    node.update({
        "leaf": True,
        "prediction": pred,
        "depth": depth,
    })

def prune_one_pass(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> int:
    '''
    parameters:
        node (Node)
        X_val (np.ndarray)
        y_val (np.ndarray)
    functionality:
        Performs one greedy pruning pass, pruning nodes only if validation error does not increase.
    return:
        int number of nodes pruned.
    '''
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
                # keep pruned
                pruned += 1
                changed = True
            else:
                # revert candidate
                cand.clear()
                cand.update(snapshot)
    return pruned

def pruning_path(node: Node, X_val: np.ndarray, y_val: np.ndarray, pass_limit: int | None = None):
    '''
    parameters:
        node (Node)
        X_val (np.ndarray)
        y_val (np.ndarray)
        pass_limit (int or None)
    functionality:
        Repeatedly prunes the tree and records the model and error after each pass until convergence or limit.
    return:
        Tuple of (list of tree copies, list of validation errors).
    '''
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
    '''
    parameters:
        node (Node)
        X_val (np.ndarray)
        y_val (np.ndarray)
        passes (int)
    functionality:
        Applies a fixed number of pruning passes or stops early if convergence is reached.
    return:
        Node after pruning.
    '''
    work = copy.deepcopy(node)
    for i in range(passes):
        n = prune_one_pass(work, X_val, y_val)
        if n == 0:
            break
    return work

def count_passes_to_converge(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> int:
    '''
    parameters:
        node (Node)
        X_val (np.ndarray)
        y_val (np.ndarray)
    functionality:
        Determines how many pruning passes occur before no more nodes can be pruned.
    return:
        int number of passes until convergence.
    '''
    models, _ = pruning_path(node, X_val, y_val)
    # number of passes is len(models)-1 (including unpruned as step 0)
    return max(0, len(models) - 1)

def max_depth_after_prune(node: Node):
    '''
    parameters:
        node (Node)
    functionality:
        Calculates the maximum depth of a (possibly pruned) decision tree.
    return:
        int maximum tree depth.
    '''
    return tree_max_depth(node)
