import pandas as pd

POS = "PHISHING"
NEG = "LEGITIMATE"


def clean_pred_label(value):
    # Clean the predicted label value, treating None and NaN as None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Convert text to uppercase and strip whitespace, if it matches POS or NEG return those
    text = str(value).strip().upper()
    if text == POS:
        return POS
    if text == NEG:
        return NEG
    return None


def clean_true_label(value):
    # Clean the true label value, treating None and NaN as None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        n = int(value)
    except Exception:
        return None

    # Convert 1 to POS and 0 to NEG
    if n == 1:
        return POS
    if n == 0:
        return NEG
    return None


def clean_type(value):
    # Clean the phishing type value, treating None, NaN and empty strings as "unknown"
    if value is None:
        return "unknown"
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "unknown"


def compute_metrics(rows):
    # Compute TP, FP, TN, FN and use them to compute accuracy, precision, recall, f1
    tp = fp = tn = fn = 0

    for row in rows:
        t = row["true"]
        p = row["pred"]

        if t == POS and p == POS:
            tp += 1
        elif t == NEG and p == POS:
            fp += 1
        elif t == NEG and p == NEG:
            tn += 1
        elif t == POS and p == NEG:
            fn += 1

    n = tp + fp + tn + fn

    # Confusion matrixed metrics with safe division to handle edge cases
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else float("nan")

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }