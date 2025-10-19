import json
import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    L = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[index[t], index[p]] += 1
    return cm

def accuracy_from_cm(cm):
    cm = np.asarray(cm)
    return float(np.trace(cm)) / float(cm.sum()) if cm.sum() else 0.0

def precision_recall_f1_from_cm(cm):
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
