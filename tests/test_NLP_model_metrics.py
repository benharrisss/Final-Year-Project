import math
import pandas as pd

import metrics_utils as mu
import NLP_model_baseline_metrics as nlp_metrics


def test_perfect_brier_is_zero():
    df = pd.DataFrame({"true": [mu.POS, mu.NEG], "predicted_prob_phishing": [1.0, 0.0]})
    brier, n = nlp_metrics.calculate_brier(df)

    # With perfect predictions, Brier score should be 0.0 and n should count both rows
    assert n == 2
    assert brier == 0.0


def test_brier_clamping():
    df = pd.DataFrame({"true": [mu.POS, mu.NEG], "predicted_prob_phishing": [1.2, -0.1]})
    brier, n = nlp_metrics.calculate_brier(df)

    # With clamping, these should be treated as perfect predictions (same as previous test)
    assert n == 2
    assert brier == 0.0


def test_brier_bad_data_is_nan():
    df = pd.DataFrame({"true": [mu.POS, mu.NEG], "predicted_prob_phishing": ["bad", float("nan")]})
    brier, n = nlp_metrics.calculate_brier(df)

    # With no usable predictions, Brier score should be NaN and n should be 0
    assert n == 0
    assert math.isnan(brier)