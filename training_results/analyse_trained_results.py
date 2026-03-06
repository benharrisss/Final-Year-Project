import argparse
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

    # Calculations for typical evaluatuion metrics
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

    has_summary_error = "summary_error" in df.columns
    has_classify_error = "classify_error" in df.columns
    has_type = "phish_type" in df.columns

    true_list = []
    pred_list = []
    type_list = []

    # Extract true, pred and phishing type and store in new columns after cleaning
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
        if i is None:
            missing_pred_count += 1

    missing_pred_rate = missing_pred_count / total_rows
    coverage = 1.0 - missing_pred_rate

    # JSON parse failed rate + summary empty count for summary step
    summary_json_fail_count = 0
    summary_failed_empty_count = 0

    if has_summary_error:
        for _, row in df.iterrows():
            # Look for errors in the summary_error column
            err = row["summary_error"]
            if err is None:
                continue
            try:
                if pd.isna(err):
                    continue
            except Exception:
                pass

            # Match "SUMMARY_JSON_PARSE_FAILED" etc. in the error text to count parse failures & empty summaries
            err_text = str(err)
            if "SUMMARY_JSON_PARSE_FAILED" in err_text:
                summary_json_fail_count += 1
            if "SUMMARY_FAILED_EMPTY" in err_text:
                summary_failed_empty_count += 1

        summary_json_fail_rate = summary_json_fail_count / total_rows
    else:
        summary_json_fail_rate = float("nan")

    # JSON parse failed rate + recovered count for classify step
    classify_json_fail_count = 0
    classify_recovered_count = 0

    if has_classify_error:
        for i, row in df.iterrows():
            # Look for errors in the classify_error column
            err = row["classify_error"]
            if err is None:
                continue
            try:
                if pd.isna(err):
                    continue
            except Exception:
                pass

            # Match "CLASSIFY_JSON_PARSE_FAILED" in the error text to count JSON parse failures
            err_text = str(err)
            if "JSON_PARSE_FAILED" in err_text:
                classify_json_fail_count += 1
                # Recovered, so parse failed but predicted label exists
                if df.at[i, "pred"] is not None:
                    classify_recovered_count += 1

        classify_json_fail_rate = classify_json_fail_count / total_rows
    else:
        classify_json_fail_rate = float("nan")

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
        # Use all usable rows that contain true, pred and a phish_type that matches the list of types
        rows_t = [r for r in usable_rows if r["phish_type"] == tname]
        m = compute_metrics(rows_t)

        # detection_rate (det_rate) = TP / (TP+FN) within that type 
        # false_positive_rate (fp_rate) = FP / (FP+TN) within that type
        det_rate = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")
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

    print("Summary and Classification error stats:")
    if has_summary_error:
        print(f"JSON Parse failures in the summary: {summary_json_fail_count} ({summary_json_fail_rate:.4f})")
        print(f"Empty summaries: {summary_failed_empty_count}")
    else:
        print("No 'summary_error' column found; skipping summary error stats.")

    if has_classify_error:
        print(f"JSON Parse failures in classification: {classify_json_fail_count} ({classify_json_fail_rate:.4f})")
        print(f"Recovered from classification errors: {classify_recovered_count}")
    else:
        print("No 'classify_error' column found; skipping classification error stats.")
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