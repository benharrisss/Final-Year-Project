import pytest


def extract_value_after_colon(stdout, prefix):
    for line in stdout.splitlines():
        line = line.strip()

        # Check if the line starts with the given prefix
        if line.startswith(prefix):
            # Take the part after the first colon and strip whitespace and use the first token as the value
            after = line.split(":", 1)[1].strip()
            first_token = after.split()[0]
            return float(first_token)

    raise AssertionError(f"Could not find line starting with: {prefix}")


def test_reproduce_llm_baseline(run_python):
    csv_path = "decoder_only_LLM_baseline/results/results_gemma-2-9b-it_2200-BL.csv"
    exec_test = run_python("metrics/LLM_baseline_metrics.py", csv_path)

    assert exec_test.returncode == 0, exec_test.stderr

    # Extract various metrics from the output
    missing_predictions = extract_value_after_colon(exec_test.stdout, "Missing predictions")
    coverage = extract_value_after_colon(exec_test.stdout, "Coverage")
    accuracy = extract_value_after_colon(exec_test.stdout, "accuracy")
    precision = extract_value_after_colon(exec_test.stdout, "precision")
    recall = extract_value_after_colon(exec_test.stdout, "recall")
    f1 = extract_value_after_colon(exec_test.stdout, "f1")
    json_parse_failed = extract_value_after_colon(exec_test.stdout, "json_parse_failed")
    recovered_count = extract_value_after_colon(exec_test.stdout, "recovered_count")

    # Assert these metrics match known values from the actual run
    assert missing_predictions == pytest.approx(25, abs=1e-4)
    assert coverage == pytest.approx(0.9886, abs=1e-4)
    assert accuracy == pytest.approx(0.9067, abs=1e-4)
    assert precision == pytest.approx(0.8584, abs=1e-4)
    assert recall == pytest.approx(0.9753, abs=1e-4)
    assert f1 == pytest.approx(0.9131, abs=1e-4)
    assert json_parse_failed == pytest.approx(26, abs=1e-4)
    assert recovered_count == pytest.approx(1, abs=1e-4)


def test_reproduce_llm_summarisation(run_python):
    csv_path = "decoder_only_LLM_enhancement/email_summarisation/results/results_Phi-mini-MoE-instruct_2200-SUM.csv"
    exec_test = run_python("metrics/LLM_enhancement_metrics.py", csv_path)

    assert exec_test.returncode == 0, exec_test.stderr

    # Extract various metrics from the output
    missing_predictions = extract_value_after_colon(exec_test.stdout, "Missing predictions")
    coverage = extract_value_after_colon(exec_test.stdout, "Coverage")
    accuracy = extract_value_after_colon(exec_test.stdout, "accuracy")
    precision = extract_value_after_colon(exec_test.stdout, "precision")
    recall = extract_value_after_colon(exec_test.stdout, "recall")
    f1 = extract_value_after_colon(exec_test.stdout, "f1")
    json_parse_failed_summary = extract_value_after_colon(exec_test.stdout, "JSON Parse failures in the summary")
    empty_summaries = extract_value_after_colon(exec_test.stdout, "Empty summaries")
    json_parse_failed_classify = extract_value_after_colon(exec_test.stdout, "JSON Parse failures in classification")
    recovered_count = extract_value_after_colon(exec_test.stdout, "Recovered from classification errors:")

    # Assert these metrics match known values from the actual run
    assert missing_predictions == pytest.approx(0, abs=1e-4)
    assert coverage == pytest.approx(1.0000, abs=1e-4)
    assert accuracy == pytest.approx(0.7073, abs=1e-4)
    assert precision == pytest.approx(0.6423, abs=1e-4)
    assert recall == pytest.approx(0.9355, abs=1e-4)
    assert f1 == pytest.approx(0.7617, abs=1e-4)
    assert json_parse_failed_summary == pytest.approx(74, abs=1e-4)
    assert empty_summaries == pytest.approx(0, abs=1e-4)
    assert json_parse_failed_classify == pytest.approx(0, abs=1e-4)
    assert recovered_count == pytest.approx(0, abs=1e-4)


def test_reproduce_nlp_baseline(run_python):
    csv_path = "encoder_only_NLP_model_baseline/results/results_secbert_660-BL.csv"
    exec_test = run_python("metrics/NLP_model_baseline_metrics.py", csv_path)

    assert exec_test.returncode == 0, exec_test.stderr

    # Extract various metrics from the output
    missing_predictions = extract_value_after_colon(exec_test.stdout, "Missing predictions")
    coverage = extract_value_after_colon(exec_test.stdout, "Coverage")
    accuracy = extract_value_after_colon(exec_test.stdout, "accuracy")
    precision = extract_value_after_colon(exec_test.stdout, "precision")
    recall = extract_value_after_colon(exec_test.stdout, "recall")
    f1 = extract_value_after_colon(exec_test.stdout, "f1")
    brier = extract_value_after_colon(exec_test.stdout, "Brier score")

    # Assert these metrics match known values from the actual run
    assert missing_predictions == pytest.approx(0, abs=1e-4)
    assert coverage == pytest.approx(1.0000, abs=1e-4)
    assert accuracy == pytest.approx(0.9530, abs=1e-4)
    assert precision == pytest.approx(0.9517, abs=1e-4)
    assert recall == pytest.approx(0.9545, abs=1e-4)
    assert f1 == pytest.approx(0.9531, abs=1e-4)
    assert brier == pytest.approx(0.0357, abs=1e-4)