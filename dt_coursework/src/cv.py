from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .tree import decision_tree_learning, predict, tree_max_depth
from .metrics import confusion_matrix, metrics_summary
from .prune import count_passes_to_converge, prune_with_passes

def kfold_indices(n, k=10, seed=42, shuffle=True):
    '''
    parameters:
        n (int): total number of samples in the dataset
        k (int): number of groups/folds to divide the dataset into
        seed (int): random seed for np.random.default_rng() to ensure reproducibility
        shuffle (bool): whether to shuffle the dataset indices before splitting

    functionality:
        Generate indices [0..n-1], shuffle if needed, and split them into k folds.

    return:
        list of tuples: each tuple contains (train_indices, test_indices)
    '''

    rng = np.random.default_rng(seed)  # initialize random number generator
    indices = np.arange(n)  # create array [0, 1, ..., n-1]

    if shuffle:
        rng.shuffle(indices)  # shuffle indices if requested

    folds = np.array_split(indices, k)  # split indices into k equal parts

    # create list of (train, test) index pairs for each fold
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]

def evaluate_tree_on_split(X_train, y_train, X_test, y_test, labels):
    '''
    parameters:
        X_train (ndarray): feature matrix of the training dataset
        y_train (ndarray): class labels for the training dataset
        X_test (ndarray): feature matrix of the test dataset
        y_test (ndarray): class labels for the test dataset
        labels (ndarray): array of all unique class labels in the dataset

    functionality:
        Train a decision tree using the training data and evaluate it on the test data.
        Compute the confusion matrix for performance evaluation.

    return:
        tuple: (trained decision tree, tree depth, confusion matrix)
    '''

    # train the decision tree and get its maximum depth
    tree, max_depth = decision_tree_learning(X_train, y_train, depth=0)

    # predict labels for the test dataset
    y_pred = predict(tree, X_test)

    # compute confusion matrix using true and predicted labels
    cm = confusion_matrix(y_test, y_pred, labels)

    return tree, max_depth, cm


def cross_validate(X, y, k=10, seed=42, prune=False, nested=False, inner_k=10):
    '''
    parameters:
        X (ndarray): feature matrix of the dataset
        y (ndarray): class labels of the dataset
        k (int): number of outer folds (train/test)
        seed (int): random seed for reproducibility
        prune (bool): whether to apply pruning
        nested (bool): whether to use nested CV to choose pruning passes
        inner_k (int): number of inner folds (train/validation) when nested is True

    functionality:
        Run k-fold CV to train/evaluate decision trees; optionally run nested CV to pick pruning passes (hyperparameter), then prune and re-evaluate.

    return:
        dict: aggregate confusion/metrics (before), depth stats, and if pruning is used, corresponding "after" results and chosen passes
    '''

    """Return aggregate metrics and depth stats.
    If prune & nested: inner CV estimates #passes to apply on outer training fold.
    """
    n = len(y)
    labels = np.unique(y.astype(int))
    outer = kfold_indices(n, k=k, seed=seed, shuffle=True)
    agg_cm = np.zeros((len(labels), len(labels)), dtype=int)
    depths_before = []
    depths_after = []
    chosen_passes = []

    for outer_i, (train_idx, test_idx) in enumerate(outer):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train full (unpruned) tree for the 'before' stats
        full_tree, depth_before, _ = evaluate_tree_on_split(X_train, y_train, X_test, y_test, labels)
        y_pred_before = predict(full_tree, X_test)
        agg_cm += confusion_matrix(y_test, y_pred_before, labels)
        depths_before.append(depth_before)

        if prune and nested:
            # Inner CV to estimate a robust number of pruning passes (median over folds)
            inner = kfold_indices(len(y_train), k=inner_k, seed=seed+outer_i+1, shuffle=True)
            inner_passes = []
            for (inner_tr_idx, inner_val_idx) in inner:
                X_in_tr, y_in_tr = X_train[inner_tr_idx], y_train[inner_tr_idx]
                X_in_val, y_in_val = X_train[inner_val_idx], y_train[inner_val_idx]
                inner_tree, _ = decision_tree_learning(X_in_tr, y_in_tr, depth=0)
                passes = count_passes_to_converge(inner_tree, X_in_val, y_in_val)
                inner_passes.append(passes)
            # Select a robust pass count (median) and apply to outer-train
            pass_limit = int(np.median(inner_passes))
            chosen_passes.append(pass_limit)

            outer_tree, _ = decision_tree_learning(X_train, y_train, depth=0)  # placeholder; refit below on sub-train
            # Use 20% of outer training as validation for reduced-error pruning
            rng = np.random.default_rng(seed + 1337 + outer_i)
            perm = rng.permutation(len(y_train))
            val_size = max(1, len(y_train)//5)
            val_idx = perm[:val_size]
            tr_idx = perm[val_size:]
            X_tr_sub, y_tr_sub = X_train[tr_idx], y_train[tr_idx]
            X_val_sub, y_val_sub = X_train[val_idx], y_train[val_idx]
            # Fit on sub-train to define structure; then prune using validation
            outer_tree, _ = decision_tree_learning(X_tr_sub, y_tr_sub, depth=0)  # intentionally replaces previous outer_tree
            pruned_tree = prune_with_passes(outer_tree, X_val_sub, y_val_sub, pass_limit)

            y_pred_after = predict(pruned_tree, X_test)
            depths_after.append(tree_max_depth(pruned_tree))
            # Post-pruning metrics will be collected in the "after" pass below.
        else:
            y_pred_after = y_pred_before
            depths_after.append(depth_before)

    summary_before = metrics_summary(agg_cm, labels)

    result: Dict[str, Any] = {
        "labels": [int(l) for l in labels],
        "confusion_before": agg_cm.tolist(),
        "metrics_before": summary_before,
        "avg_depth_before": float(np.mean(depths_before)),
        "std_depth_before": float(np.std(depths_before)),
    }

    if prune and nested:
        # Second pass: compute "after" metrics with pruning using same outer folds
        agg_cm_after = np.zeros_like(agg_cm)
        depths_after = []
        chosen_passes = []

        for outer_i, (train_idx, test_idx) in enumerate(outer):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Inner CV again (same seed path -> deterministic choice)
            inner = kfold_indices(len(y_train), k=inner_k, seed=seed+outer_i+1, shuffle=True)
            inner_passes = []
            for (inner_tr_idx, inner_val_idx) in inner:
                X_in_tr, y_in_tr = X_train[inner_tr_idx], y_train[inner_tr_idx]
                X_in_val, y_in_val = X_train[inner_val_idx], y_train[inner_val_idx]
                inner_tree, _ = decision_tree_learning(X_in_tr, y_in_tr, depth=0)
                passes = count_passes_to_converge(inner_tree, X_in_val, y_in_val)
                inner_passes.append(passes)
            pass_limit = int(np.median(inner_passes))
            chosen_passes.append(pass_limit)

            # Build outer tree on sub-train (80%) and prune on 20% validation
            rng = np.random.default_rng(seed + 1337 + outer_i)
            perm = rng.permutation(len(y_train))
            val_size = max(1, len(y_train)//5)
            val_idx = perm[:val_size]
            tr_idx = perm[val_size:]

            X_tr_sub, y_tr_sub = X_train[tr_idx], y_train[tr_idx]
            X_val_sub, y_val_sub = X_train[val_idx], y_train[val_idx]

            outer_tree, _ = decision_tree_learning(X_tr_sub, y_tr_sub, depth=0)
            pruned_tree = prune_with_passes(outer_tree, X_val_sub, y_val_sub, pass_limit)

            y_pred = predict(pruned_tree, X_test)
            agg_cm_after += confusion_matrix(y_test, y_pred, labels)
            depths_after.append(tree_max_depth(pruned_tree))

        summary_after = metrics_summary(agg_cm_after, labels)

        result.update({
            "confusion_after": agg_cm_after.tolist(),
            "metrics_after": summary_after,
            "avg_depth_after": float(np.mean(depths_after)),
            "std_depth_after": float(np.std(depths_after)),
            "median_chosen_passes": int(np.median(np.array(chosen_passes)) if chosen_passes else 0),
        })

    return result