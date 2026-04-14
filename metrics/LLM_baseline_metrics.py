import argparse
import pandas as pd
from metrics_utils import POS, NEG, clean_true_label, clean_pred_label, clean_type, compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to baseline results CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Needs to include true and predicted labels
    required = ["true_label", "predicted_label"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"CSV must contain column: {col}")

    has_error = "error" in df.columns
    has_type = "phish_type" in df.columns

    true_list = []
    pred_list = []
    type_list = []

    # Clean/normalise values before computing metrics
    for _, row in df.iterrows():
        true_list.append(clean_true_label(row["true_label"]))
        pred_list.append(clean_pred_label(row["predicted_label"]))
        if has_type:
            type_list.append(clean_type(row["phish_type"]))
        else:
            type_list.append("unknown")

    df["true"] = true_list
    df["pred"] = pred_list
    df["phish_type"] = type_list

    total_rows = len(df)
    if total_rows == 0:
        raise SystemExit("CSV has no rows.")

    # Count missing predictions and calculate coverage
    missing_pred_count = 0

    for i in df["pred"]:
        if i is None or pd.isna(i):
            missing_pred_count += 1

    missing_pred_rate = missing_pred_count / total_rows

    # Coverage treats missing/invalid predictions as non-usable rows
    coverage = 1.0 - missing_pred_rate

    # JSON parse failure stats - count failures based on error tags
    if has_error:
        json_fail_count = 0
        recovered_count = 0

        for i, row in df.iterrows():
            # Look for errors in the error column
            err = row["error"]
            if err is None:
                continue
            try:
                if pd.isna(err):
                    continue
            except Exception:
                pass

            # Match "JSON_PARSE_FAILED" error tag to count parse failure
            err_text = str(err)
            if "JSON_PARSE_FAILED" in err_text:
                json_fail_count += 1
                # Recovered, so parse failed but predicted label exists
                if df.at[i, "pred"] == "PHISHING" or df.at[i, "pred"] == "LEGITIMATE":
                    recovered_count += 1

        json_fail_rate = json_fail_count / total_rows
    else:
        json_fail_count = 0
        recovered_count = 0
        json_fail_rate = float("nan")

    # Only evaluate rows where we have both true and pred
    usable_rows = []
    for _, row in df.iterrows():
        if row["true"] is None:
            continue
        if row["pred"] is None:
            continue
        usable_rows.append(row)

    overall = compute_metrics(usable_rows)

    # Per phish_type stats
    per_type = []
    types = sorted(set(df["phish_type"]))

    for tname in types:
        rows_t = [r for r in usable_rows if r["phish_type"] == tname]
        m = compute_metrics(rows_t)

        # detection_rate is recall within that type
        det_rate = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")
        
        # false_positive_rate is FPR within that type
        fp_rate = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else float("nan")

        per_type.append(
            {
                "phish_type": tname,
                "n": m["n"],
                "accuracy": m["accuracy"],
                "detection_rate (phishing)": det_rate,
                "false_positive_rate (legitimate)": fp_rate,
            }
        )

    # Sort by amount of usable rows for that type
    per_type_df = pd.DataFrame(per_type)
    if "n" in per_type_df.columns:
        per_type_df = per_type_df.sort_values(by=["n"], ascending=[False])

    # Print results
    print(f"File: {args.csv}")
    print(f"Rows: {total_rows}")
    print(f"Missing predictions: {missing_pred_count} ({missing_pred_rate:.4f})")
    print(f"Coverage: {coverage:.4f}")
    print()

    print("Overall binary metrics - (positive = PHISHING), (negative = LEGITIMATE):")
    print(f"Usable rows (true+pred): {overall['n']}")
    print(f"TP={overall['tp']}  FP={overall['fp']}  TN={overall['tn']}  FN={overall['fn']}")
    print(f"accuracy:  {overall['accuracy']:.4f}")
    print(f"precision: {overall['precision']:.4f}")
    print(f"recall:    {overall['recall']:.4f}")
    print(f"f1:        {overall['f1']:.4f}")
    print()

    print("Parsing stats:")
    if has_error:
        print(f"json_parse_failed: {json_fail_count} ({json_fail_rate:.4f})")
        print(f"recovered_count (parse failed but prediction exists): {recovered_count}")
    else:
        print("No 'error' column found; skipping JSON parse stats.")
    print()

    print("Per phish_type stats:")
    cols = [
        "phish_type",
        "n",
        "accuracy",
        "detection_rate (phishing)",
        "false_positive_rate (legitimate)",
    ]
    print(per_type_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()