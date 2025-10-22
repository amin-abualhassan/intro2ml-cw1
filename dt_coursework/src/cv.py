from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .tree import decision_tree_learning, predict, tree_max_depth
from .metrics import confusion_matrix, metrics_summary
from .prune import count_passes_to_converge, prune_with_passes

def kfold_indices(n, k=10, seed=42, shuffle=True):
    '''
    Input:
        n: total number of samples in the dataset
        k: the number of groups/folds into which the dataset is divided
        seed: starting state for generating random numbers of np.random.default_rng(), ensuring a reproducible randomized dataset sequence
        shuffle: boolean of indicator to shuffle the array of dataset indices 
    Process:
        Generate a list of random sequence of number/indices [0..n-1] 
        Split the list into k-number of equal-sized sublist
    Return:
        List of tupels containing k training and test validation indices 
    '''

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, k)

    '''
    return is list of tuples contain:
        1. list of indices of training dataset (n*(k-1)/k number of indices)
        2. list of indices of test/validation dataset (n/k number of indices)
    '''
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]

def evaluate_tree_on_split(X_train, y_train, X_test, y_test, labels):
    '''
    Input:
        X_train: Array of features of the train dataset
        y_train: Array of class of the train dataset
        X_test: Array of features of the test dataset
        y_test: Array of class of the test dataset
        labels: Array of all unique class labels in the dataset
    Process:
        Create a decision tree from training dataset (X_train and y_train)
        Evaluate the decision tree using the test dataset (X_test and y_test)
        Calculate a confusion matrix
    Return:
        tuple of:
            1. Decision tree model
            2. Decision tree depth
            3. Confusion matrix of the test dataset
    '''

    tree, max_depth = decision_tree_learning(X_train, y_train, depth=0)
    y_pred = predict(tree, X_test)
    cm = confusion_matrix(y_test, y_pred, labels)
    return tree, max_depth, cm

def cross_validate(X, y, k=10, seed=42, prune=False, nested=False, inner_k=10):
    '''
    Input:
        X: Array of features of the dataset
        y: Array of class of the dataset
        k: the number of groups/folds into which the dataset is divided (into training and test dataset)
        seed: starting state for generating random numbers of np.random.default_rng(), ensuring a reproducible randomized dataset sequence
        prune and nested: boolean of indicator to evaluate the k-fold decision tree pruning
        inner_k: the number of groups/folds into which the training dataset is divided (into training and validation dataset) 
    Process:
        1. Do k-number of decision tree learning using n*(k-1)/k training dataset & evaluate
        2. Evaluate each decision tree using n/k number of test dataset
        3. If prune and nested is true, do the pruning cross validation (k * (k-1) folds) to result the best hyperparameter (number of pruned node in the tree).
    Return:
        Evaluation result of the cross validation and and evaluation result of the pruning cross validation
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
            # Inner CV to estimate number of pruning passes
            '''
            Dataset splitted into:
                - n*(k-2)/k training dataset
                - n/k validation dataset
                - n/k training dataset
            Choosen hyperparameter: median of number of pruning which resulting to the best accuracy in each fold            
            '''
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

            outer_tree, _ = decision_tree_learning(X_train, y_train, depth=0)  ######---->>> WHAT"S THIS FOR?
            # Use 20% of outer training as validation to perform the passes
            # (reduced-error pruning requires a validation set to measure error during each pass)
            # We create a stratified-ish split by shuffling indices.
            rng = np.random.default_rng(seed + 1337 + outer_i)
            perm = rng.permutation(len(y_train))
            val_size = max(1, len(y_train)//5)
            val_idx = perm[:val_size]
            tr_idx = perm[val_size:]
            X_tr_sub, y_tr_sub = X_train[tr_idx], y_train[tr_idx]
            X_val_sub, y_val_sub = X_train[val_idx], y_train[val_idx]
            # Fit on the sub-train used to define node counts and structure 
            outer_tree, _ = decision_tree_learning(X_tr_sub, y_tr_sub, depth=0) ######---->>> THE PREVIOUS outer_tree GOT REPLACED BEFORE EVEN BEING USED
            # Apply chosen number of passes measured on X_val_sub
            pruned_tree = prune_with_passes(outer_tree, X_val_sub, y_val_sub, pass_limit)

            y_pred_after = predict(pruned_tree, X_test)
            depths_after.append(tree_max_depth(pruned_tree))
            # For 'after', we overwrite the confusion addition we already made with unpruned?
            # The spec wants post-pruning metrics separately, so we'll return separately.
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
        # Need a second pass to compute AFTER metrics with pruning using the same outer folds
        agg_cm_after = np.zeros_like(agg_cm)
        depths_after = []
        chosen_passes = []

        for outer_i, (train_idx, test_idx) in enumerate(outer):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Inner CV again (same seed path -> same choice deterministically)
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
