import math

import pandas as pd

import metrics_utils as mu


def test_clean_pred_label():
    # Valid responses
    assert mu.clean_pred_label("PHISHING") == mu.POS
    assert mu.clean_pred_label("LEGITIMATE") == mu.NEG

    # Invalid/missing responses
    assert mu.clean_pred_label(None) is None
    assert mu.clean_pred_label(float("nan")) is None
    assert mu.clean_pred_label("") is None
    assert mu.clean_pred_label("0") is None
    assert mu.clean_pred_label("PHISH") is None
    assert mu.clean_pred_label("LEGIT") is None

def test_clean_true_label():
    # Valid responses
    assert mu.clean_true_label(1) == mu.POS
    assert mu.clean_true_label("1") == mu.POS
    assert mu.clean_true_label(0) == mu.NEG
    assert mu.clean_true_label("0") == mu.NEG

    # Invalid/missing responses
    assert mu.clean_true_label(None) is None
    assert mu.clean_true_label(float("nan")) is None
    assert mu.clean_true_label(2) is None
    assert mu.clean_true_label(-1) is None
    assert mu.clean_true_label(10) is None
    assert mu.clean_true_label("phishing") is None
    assert mu.clean_true_label("legitimate") is None


def test_clean_type():
    # Valid responses
    assert mu.clean_type("invoice_payment") == "invoice_payment"

    # Invalid/missing responses
    assert mu.clean_type(None) == "unknown"
    assert mu.clean_type(float("nan")) == "unknown"
    assert mu.clean_type("") == "unknown"


def test_compute_metrics_known():
    # Build exactly 4 examples: TP=1, FP=1, TN=1, FN=1
    rows = [
        {"true": mu.POS, "pred": mu.POS, "phish_type": "invoice_payment"},
        {"true": mu.NEG, "pred": mu.POS, "phish_type": "invoice_payment"},
        {"true": mu.NEG, "pred": mu.NEG, "phish_type": "invoice_payment"},
        {"true": mu.POS, "pred": mu.NEG, "phish_type": "invoice_payment"},
    ]
    m = mu.compute_metrics(rows)

    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["fn"] == 1
    assert m["n"] == 4

    # All metrics should be 0.5 with this setup
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_compute_metrics_empty_is_nan():
    m = mu.compute_metrics([])

    assert m["n"] == 0

    # With no usable rows, all metrics should be NaN
    assert math.isnan(m["accuracy"])
    assert math.isnan(m["precision"])
    assert math.isnan(m["recall"])
    assert math.isnan(m["f1"])