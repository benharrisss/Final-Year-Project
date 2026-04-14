import json
import pandas as pd


def test_split_train_test_valid(tmp_path, run_python):
    input_csv = tmp_path / "input.csv"
    train1 = tmp_path / "train1.csv"
    test1 = tmp_path / "test1.csv"
    train2 = tmp_path / "train2.csv"
    test2 = tmp_path / "test2.csv"

    # 10 phish, 10 legit with unique ids
    df = pd.DataFrame([{"id": i, "label": 1} for i in range(10)] + [{"id": i + 44, "label": 0} for i in range(10)])
    df.to_csv(input_csv, index=False)

    exec_test1 = run_python(
        "data/datasets_for_fine-tuning/scripts/split_train_test.py",
        "--input-csv", input_csv,
        "--train-out-csv", train1,
        "--test-out-csv", test1,
        "--train-phish", 5,
        "--train-legit", 5,
    )
    assert exec_test1.returncode == 0, exec_test1.stderr

    exec_test2 = run_python(
        "data/datasets_for_fine-tuning/scripts/split_train_test.py",
        "--input-csv", input_csv,
        "--train-out-csv", train2,
        "--test-out-csv", test2,
        "--train-phish", 5,
        "--train-legit", 5,
    )
    assert exec_test2.returncode == 0, exec_test2.stderr

    train_df = pd.read_csv(train1)
    test_df = pd.read_csv(test1)

    # Data leakage check, make sure no id appears in both train and test sets
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))

    # Determinism check, make sure same train ids are selected across runs (fixed random state)
    assert set(pd.read_csv(train1)["id"]) == set(pd.read_csv(train2)["id"])

    # Train size check, make sure we got the requested number of phish and legit in the train set
    assert len(train_df) == 10
    assert len(test_df) == 10


def test_split_train_test_missing_label_fails(tmp_path, run_python):
    input_csv = tmp_path / "input.csv"
    train_out = tmp_path / "train.csv"
    test_out = tmp_path / "test.csv"

    # Missing label column should cause the script to fail
    pd.DataFrame({"id": [1, 2, 3], "subject": ["a", "b", "c"]}).to_csv(input_csv, index=False)

    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/split_train_test.py",
        "--input-csv", input_csv,
        "--train-out-csv", train_out,
        "--test-out-csv", test_out,
        "--train-phish", 1,
        "--train-legit", 1,
    )

    assert exec_test.returncode != 0


def test_split_train_test_not_enough_rows_fails(tmp_path, run_python):
    input_csv = tmp_path / "input.csv"
    train_out = tmp_path / "train.csv"
    test_out = tmp_path / "test.csv"

    # Only 2 phish + 2 legit available
    df = pd.DataFrame([{"id": i, "label": 1} for i in range(2)] + [{"id": i + 44, "label": 0} for i in range(2)])
    df.to_csv(input_csv, index=False)

    # Ask for more than exists in the dataset, should cause the script to fail
    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/split_train_test.py",
        "--input-csv", input_csv,
        "--train-out-csv", train_out,
        "--test-out-csv", test_out,
        "--train-phish", 5,
        "--train-legit", 5,
    )

    assert exec_test.returncode != 0


def test_build_fewshots_valid(tmp_path, run_python):
    input_csv = tmp_path / "train_summaries.csv"
    output = tmp_path / "shots.jsonl"

    pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4", "s5", "s6"],
            # Include one empty summary to test that filtering works correctly
            "summary": ["ok", "ok", "ok", "ok", "ok", ""],
            "true_label": [1, 1, 0, 0, 1, 0],
        }
    ).to_csv(input_csv, index=False)

    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/build_fewshots.py",
        "--train-summaries-csv", input_csv,
        "--train-jsonl", output,
        "--num-phish", 2,
        "--num-legit", 2,
    )

    assert exec_test.returncode == 0, exec_test.stderr

    lines = output.read_text(encoding="utf-8").strip().splitlines()

    # Should have 4 lines (2 phish + 2 legit)
    assert len(lines) == 4

    for line in lines:
        obj = json.loads(line)

        # Check that the label is either PHISHING or LEGITIMATE
        assert obj["label"] in ["PHISHING", "LEGITIMATE"]

        # Check that the summary is not empty (should have filtered out the empty one)
        assert obj["summary"].strip() != ""


def test_build_fewshots_missing_subject_fails(tmp_path, run_python):
    input_csv = tmp_path / "train_summaries.csv"
    output = tmp_path / "shots.jsonl"

    # Missing subject column should cause the script to fail
    pd.DataFrame({"summary": ["ok", "ok"], "true_label": [1, 0]}).to_csv(input_csv, index=False)

    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/build_fewshots.py",
        "--train-summaries-csv", input_csv,
        "--train-jsonl", output,
        "--num-phish", 1,
        "--num-legit", 1,
    )

    assert exec_test.returncode != 0


def test_build_fewshots_missing_summary_fails(tmp_path, run_python):
    input_csv = tmp_path / "train_summaries.csv"
    output = tmp_path / "shots.jsonl"

    # Missing summary column should cause the script to fail
    pd.DataFrame({"subject": ["s1", "s2"], "true_label": [1, 0]}).to_csv(input_csv, index=False)

    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/build_fewshots.py",
        "--train-summaries-csv", input_csv,
        "--train-jsonl", output,
        "--num-phish", 1,
        "--num-legit", 1,
    )

    assert exec_test.returncode != 0


def test_build_fewshots_missing_true_label_fails(tmp_path, run_python):
    input_csv = tmp_path / "train_summaries.csv"
    output = tmp_path / "shots.jsonl"

    # Missing true_label column should cause the script to fail
    pd.DataFrame({"subject": ["s1", "s2"], "summary": ["ok", "ok"]}).to_csv(input_csv, index=False)

    exec_test = run_python(
        "data/datasets_for_fine-tuning/scripts/build_fewshots.py",
        "--train-summaries-csv", input_csv,
        "--train-jsonl", output,
        "--num-phish", 1,
        "--num-legit", 1,
    )

    assert exec_test.returncode != 0