from __future__ import annotations
import copy
import numpy as np
from typing import Tuple, Dict, Any, List
from .tree import predict, tree_max_depth
from .utils import majority_label

Node = Dict[str, Any]

def _iter_prunable_nodes(node: Node, parent=None, side=None):
    """
    Input:
        node (Node): Current Node the decision tree.
        parent (Node, optional): Parent node of the current node.
        side (str, optional): Specifies if the node is the Left or the Right Child.

    Process:
        Traverses the tree recursively and identifies all the nodes who's left and right children are
        leaf nodes.

    Return:
        Tuple of (node, parent, side) for each node that can be pruned.
    """
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
    """
    Input:
        node (Node): Root node of the decision tree.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): True labels for the validation data.

    Process:
        Predicts labels for the validation set using the given decision tree
        and compares them to the true labels.

    Return:
        float: The proportion of incorrect predictions (validation error rate).
    """
    if X_val.size == 0:
        return 0.0
    preds = predict(node, X_val)
    return float(np.mean(preds != y_val))

def _prune_node_in_place(node: Node):
    """
    Input:
        node (Node): A non-leaf node that is a candidate for pruning.

    Process:
        Replaces the given node with a single leaf node that predicts the majority
        class (most common label) of the samples under that node.
        This effectively removes its left and right children

    Return:
         None — the node is modified directly to become a leaf node.
    """
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
    """
    Input:
        node (Node): The full decision tree to prune.
        X_val (np.ndarray): Validation dataset features.
        y_val (np.ndarray): Validation labels.

    Process:
        Performs one greedy pruning pass:
        - Finds nodes where both children are leaves.
        - Temporarily prunes each candidate node.
        - Keeps the pruning only if the validation error does not increase.
        - Reverts the node if pruning decreases the validation accuracy.
        Repeats this process until no more nodes can be pruned.

    Return:
        int: Number of nodes successfully pruned during this pass.
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
                # keep pruned
                pruned += 1
                changed = True
            else:
                # revert candidate
                cand.clear()
                cand.update(snapshot)
    return pruned

def pruning_path(node: Node, X_val: np.ndarray, y_val: np.ndarray, pass_limit: int | None = None):
    """
    Input:
        node (Node): The original unpruned decision tree.
        X_val (np.ndarray): Validation dataset features.
        y_val (np.ndarray): Validation labels.
        pass_limit (int, optional): Maximum number of pruning passes allowed.

    Process:
        Repeatedly applies pruning passes one by one.
        After each pass, stores the resulting tree and its validation error.
        Stops if pruning converges (no nodes pruned) or if the pass limit is reached.

    Return:
        Tuple[List[Node], List[float]]:
            - models: List of tree copies after each pruning pass.
            - errors: Corresponding list of validation errors for each version.
    """
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
    """
    Input:
        node (Node): The original decision tree.
        X_val (np.ndarray): Validation dataset features.
        y_val (np.ndarray): Validation labels.
        passes (int): Number of pruning passes to perform.

    Process:
        Runs a fixed number of pruning passes (or fewer if convergence occurs earlier).
        In each pass, prunes the tree greedily based on validation error.

    Return:
        Node: The pruned decision tree after the specified number of passes.
    """
    work = copy.deepcopy(node)
    for i in range(passes):
        n = prune_one_pass(work, X_val, y_val)
        if n == 0:
            break
    return work

def count_passes_to_converge(node: Node, X_val: np.ndarray, y_val: np.ndarray) -> int:
    """
    Input:
        node (Node): The original decision tree.
        X_val (np.ndarray): Validation dataset features.
        y_val (np.ndarray): Validation labels.

    Process:
        Tracks how many pruning passes are needed before the tree stops changing. This means
        that no more nodes can be pruned without increasing validation error.

    Return:
        int: Number of passes required for pruning to converge.
    """
    models, _ = pruning_path(node, X_val, y_val)
    # number of passes is len(models)-1 (including unpruned as step 0)
    return max(0, len(models) - 1)

def max_depth_after_prune(node: Node):
    """
    Input:
        node (Node): The pruned or unpruned decision tree.

    Process:
        Traverses the tree to calculate its maximum depth.

    Return:
        int: Maximum depth of the tree after pruning.
    """
    return tree_max_depth(node)
