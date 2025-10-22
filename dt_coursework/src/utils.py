import numpy as np

def _coerce_to_int_labels(y):
    '''
    parameters:

    y (array-like)

    functionality:
    Return a 1D int array of labels. Flattens input, drops NaNs, accepts numeric strings.
    return:
    np.ndarray (int)
    '''
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()  # ensure 1D

    if y.size == 0:
        return np.array([], dtype=int)

    if np.issubdtype(y.dtype, np.floating):
        y = y[~np.isnan(y)]  # drop NaNs before casting
        if y.size == 0:
            return np.array([], dtype=int)

    try:
        return y.astype(int)  # fast path
    except (ValueError, TypeError):
        try:
            return y.astype(float).astype(int)  # handle numeric strings like "3" or "3.0"
        except Exception as e:
            # clear error message for non-numeric labels
            raise TypeError("Labels must be numeric (or numeric strings).") from e


def unique_labels(y):
    '''
    parameters:

    y (array-like)

    functionality:
    Return sorted unique labels after robust int coercion.
    return:
    np.ndarray (int)
    '''
    y_int = _coerce_to_int_labels(y)  # normalize labels
    return np.unique(y_int)


def majority_label(y):
    '''
    parameters:

    y (array-like)

    functionality:
    Return the most frequent label after int coercion. Ties go to the smallest label.
    return:
    int
    '''
    y_int = _coerce_to_int_labels(y)
    if y_int.size == 0:
        raise ValueError("majority_label: empty label array after coercion.")
    vals, counts = np.unique(y_int, return_counts=True)  # sorted labels + counts
    return int(vals[np.argmax(counts)])  # argmax on counts; ties pick first (smallest label)


def label_counts(y, labels=None):
    '''
    parameters:

    y (array-like), labels (array-like or None)

    functionality:
    Count how often each label in `labels` appears in y (after int coercion). If labels is None, use unique_labels(y).
    return:
    np.ndarray (int) aligned with `labels`
    '''
    y_int = _coerce_to_int_labels(y)

    if labels is None:
        labels_int = unique_labels(y_int)  # derive label order
    else:
        labels_arr = np.asarray(labels)
        try:
            labels_int = labels_arr.astype(int)
        except (ValueError, TypeError):
            try:
                labels_int = labels_arr.astype(float).astype(int)  # allow numeric strings
            except Exception as e:
                raise TypeError("label_counts: `labels` must be numeric (or numeric strings).") from e

    # count occurrences in y_int
    vals, cnts = np.unique(y_int, return_counts=True)
    idx = {int(v): i for i, v in enumerate(vals.tolist())}  # label -> position in vals
    out = np.zeros(len(labels_int), dtype=int)
    for j, lab in enumerate(labels_int.tolist()):
        i = idx.get(int(lab), None)
        if i is not None:
            out[j] = int(cnts[i])  # align counts to requested label order
    return out


def entropy_from_counts(counts):
    '''
    parameters:

    counts (array-like of nonnegative numbers)

    functionality:
    Compute Shannon entropy (base 2). Ignores zero counts. Returns 0 if total is 0.
    return:
    float
    '''
    c = np.asarray(counts, dtype=float).ravel()  # 1D float array
    if c.size == 0:
        return 0.0
    if np.any(c < 0):
        raise ValueError("entropy_from_counts: counts must be nonnegative.")
    n = c.sum()
    if n == 0:
        return 0.0
    p = c / n
    p = p[p > 0]  # avoid log2(0)
    return float(-(p * np.log2(p)).sum())


def information_gain(parent_counts, left_counts, right_counts):
    '''
    parameters:

    parent_counts, left_counts, right_counts (array-like of nonnegative numbers)

    functionality:
    Information gain: H(parent) minus weighted child entropies. Returns 0 if a child is empty.
    Uses child totals for weights if sums differ. Clips tiny negative IG to 0.
    return:
    float
    '''
    parent = np.asarray(parent_counts, dtype=float).ravel()
    left   = np.asarray(left_counts,   dtype=float).ravel()
    right  = np.asarray(right_counts,  dtype=float).ravel()

    if np.any(parent < 0) or np.any(left < 0) or np.any(right < 0):
        raise ValueError("information_gain: counts must be nonnegative.")

    n_parent = parent.sum()
    n_left   = left.sum()
    n_right  = right.sum()

    if n_left == 0 or n_right == 0:
        return 0.0  # degenerate split

    H_parent = entropy_from_counts(parent)
    H_left   = entropy_from_counts(left)
    H_right  = entropy_from_counts(right)

    # prefer child totals for weights if sums don't match exactly
    n_weight = n_left + n_right if not np.isclose(n_parent, n_left + n_right) else n_parent
    if n_weight == 0:
        return 0.0

    remainder = (n_left / n_weight) * H_left + (n_right / n_weight) * H_right
    ig = H_parent - remainder

    # guard against tiny negative due to floating point
    if ig < 0 and ig > -1e-12:
        ig = 0.0
    return float(ig)
