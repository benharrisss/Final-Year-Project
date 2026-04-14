import pandas as pd


def test_llm_baseline_metrics_valid(tmp_path, run_python):
    csv_path = tmp_path / "test.csv"

    # Create a test CSV with all necessary columns and some example data
    pd.DataFrame(
        {
            "true_label": [1, 0],
            "predicted_label": ["PHISHING", "LEGITIMATE"],
            "phish_type": ["hr_impersonation", "loans"],
            "error": [None, "JSON_PARSE_FAILED"],
        }
    ).to_csv(csv_path, index=False)

    # Run the LLM baseline metrics script on the test CSV
    exec_test = run_python("metrics/LLM_baseline_metrics.py", csv_path)

    # Check that the script ran successfully and printed expected output sections
    assert exec_test.returncode == 0, exec_test.stderr
    assert "Coverage:" in exec_test.stdout
    assert "TP=" in exec_test.stdout
    assert "accuracy:" in exec_test.stdout  
    assert "Parsing stats:" in exec_test.stdout
    assert "json_parse_failed" in exec_test.stdout
    assert "Per phish_type stats:" in exec_test.stdout


def test_llm_baseline_metrics_missing_true_label_fails(tmp_path, run_python):
    csv_path = tmp_path / "bad.csv"

    # Missing true_label column should cause the script to fail
    pd.DataFrame({"predicted_label": ["PHISHING", "LEGITIMATE"]}).to_csv(csv_path, index=False)

    exec_test = run_python("metrics/LLM_baseline_metrics.py", csv_path)

    assert exec_test.returncode != 0


def test_llm_baseline_metrics_missing_pred_label_fails(tmp_path, run_python):
    csv_path = tmp_path / "bad.csv"

    # Missing predicted_label column should cause the script to fail
    pd.DataFrame({"true_label": [1, 0]}).to_csv(csv_path, index=False)

    exec_test = run_python("metrics/LLM_baseline_metrics.py", csv_path)

    assert exec_test.returncode != 0


def test_llm_enhancement_metrics_valid(tmp_path, run_python):
    csv_path = tmp_path / "test.csv"

    # Create a test CSV with all necessary columns and some example data (with summary and classify errors)
    pd.DataFrame(
        {
            "true_label": [1, 0],
            "predicted_label": ["PHISHING", "LEGITIMATE"],
            "phish_type": ["hr_impersonation", "loans"],
            "summary_error": ["SUMMARY_JSON_PARSE_FAILED", None],
            "classify_error": ["JSON_PARSE_FAILED", None],
        }
    ).to_csv(csv_path, index=False)

    # Run the LLM enhancement metrics script on the test CSV
    exec_test = run_python("metrics/LLM_enhancement_metrics.py", csv_path)

    # Check that the script ran successfully and printed expected output sections
    assert exec_test.returncode == 0, exec_test.stderr
    assert "Coverage:" in exec_test.stdout
    assert "TP=" in exec_test.stdout
    assert "accuracy:" in exec_test.stdout
    assert "Summary and Classification error stats:" in exec_test.stdout
    assert "JSON Parse failures in the summary:" in exec_test.stdout
    assert "JSON Parse failures in classification:" in exec_test.stdout
    assert "Per phish_type stats:" in exec_test.stdout


def test_llm_enhancement_metrics_missing_true_label_fails(tmp_path, run_python):
    csv_path = tmp_path / "bad.csv"

    # Missing true_label column should cause the script to fail
    pd.DataFrame({"predicted_label": ["PHISHING", "LEGITIMATE"]}).to_csv(csv_path, index=False)

    exec_test = run_python("metrics/LLM_enhancement_metrics.py", csv_path)

    assert exec_test.returncode != 0


def test_llm_enhancement_metrics_missing_pred_label_fails(tmp_path, run_python):
    csv_path = tmp_path / "bad.csv"

    # Missing predicted_label column should cause the script to fail
    pd.DataFrame({"true_label": [1, 0]}).to_csv(csv_path, index=False)

    exec_test = run_python("metrics/LLM_enhancement_metrics.py", csv_path)

    assert exec_test.returncode != 0