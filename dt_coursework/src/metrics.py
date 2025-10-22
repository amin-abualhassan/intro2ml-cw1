import json
import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    """
    Input:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        labels (list or np.ndarray): List of all possible class labels.

    Process:
        - Initializes an empty LxL confusion matrix, where L = number of labels.
        - Iterates over each true and predicted pair.
        - Increments the matrix cell corresponding to (true_label, predicted_label).

    Return:
        np.ndarray: A 2D confusion matrix where rows represent true labels and
                    columns represent predicted labels.
    """
    L = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[index[t], index[p]] += 1
    return cm

def accuracy_from_cm(cm):
    """
    Input:
        cm (np.ndarray): Confusion matrix.

    Process:
        - Calculates accuracy as (sum of diagonal elements) / (total samples).

    Return:
        float: Overall classification accuracy.
    """
    cm = np.asarray(cm)
    return float(np.trace(cm)) / float(cm.sum()) if cm.sum() else 0.0

def precision_recall_f1_from_cm(cm):
    """
    Input:
        cm (np.ndarray): Confusion matrix.

    Process:
        - For each class (i):
            * True Positives (TP): cm[i, i]
            * False Positives (FP): sum of column i minus TP
            * False Negatives (FN): sum of row i minus TP
        - Computes:
            * Precision = TP / (TP + FP)
            * Recall = TP / (TP + FN)
            * F1-score = 2 * (Precision * Recall) / (Precision + Recall)
        - Handles division by zero cases gracefully.

    Return:
        tuple: (precisions, recalls, f1s)
            - precisions (np.ndarray): precision values per class.
            - recalls (np.ndarray): recall values per class.
            - f1s (np.ndarray): F1 scores per class.
    """
    cm = np.asarray(cm)
    L = cm.shape[0]
    precisions = np.zeros(L)
    recalls = np.zeros(L)
    f1s = np.zeros(L)
    for i in range(L):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions[i] = prec
        recalls[i] = rec
        f1s[i] = f1
    return precisions, recalls, f1s

def metrics_summary(cm, labels):
    """
    Input:
        cm (np.ndarray): Confusion matrix.
        labels (list or np.ndarray): List of all possible class labels.

    Process:
        - Computes overall accuracy using the confusion matrix.
        - Computes precision, recall, and F1-score for each class.
        - Packages all metrics into a dictionary with readable structure.

    Return:
        dict: Summary containing:
            * 'labels': list of label IDs
            * 'confusion_matrix': confusion matrix as a nested list
            * 'accuracy': overall accuracy
            * 'per_class': dictionary of precision, recall, and F1 for each class
    """
    acc = accuracy_from_cm(cm)
    prec, rec, f1 = precision_recall_f1_from_cm(cm)
    return {
        "labels": [int(l) for l in labels],
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "per_class": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist()
        }
    }

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
