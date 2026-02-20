import argparse
import pandas as pd

POS = "PHISHING"
NEG = "LEGITIMATE"


def clean_pred_label(value):
    """Return POS/NEG or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip().upper()
    if text == POS:
        return POS
    if text == NEG:
        return NEG
    return None


def clean_true_label(value):
    """Return POS/NEG or None (but your data should always be 0/1)."""
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

    if n == 1:
        return POS
    if n == 0:
        return NEG
    return None


def clean_type(value):
    if value is None:
        return "unknown"
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to baseline results CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = ["true_label", "predicted_label"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"CSV must contain column: {col}")

    has_error = "error" in df.columns
    has_type = "phish_type" in df.columns

    true_list = []
    pred_list = []
    type_list = []

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

    missing_pred_count = sum(1 for x in df["pred"] if x is None)
    missing_pred_rate = missing_pred_count / total_rows

    # With assumption that true labels are always valid 0/1:
    coverage = 1.0 - missing_pred_rate

    # JSON parse failed rate + recovered count
    if has_error:
        json_fail_count = 0
        recovered_count = 0

        for i, row in df.iterrows():
            err = row["error"]
            if err is None:
                continue
            try:
                if pd.isna(err):
                    continue
            except Exception:
                pass

            err_text = str(err)
            if "JSON_PARSE_FAILED" in err_text:
                json_fail_count += 1
                # "Recovered" - parse failed but predicted label exists
                if df.at[i, "pred"] is not None:
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

    # Overall confusion matrix (positive = PHISHING)
    tp = fp = tn = fn = 0
    for row in usable_rows:
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

    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else float("nan")


    # detection_rate_if_phish = TP / (TP+FN) within that type 
    # false_positive_rate_if_legit = FP / (FP+TN) within that type
    per_type = []
    types = sorted(set(df["phish_type"]))

    for tname in types:
        # rows for this type where true+pred exist
        rows_t = []
        for _, row in df.iterrows():
            if row["phish_type"] != tname:
                continue
            if row["true"] is None or row["pred"] is None:
                continue
            rows_t.append(row)

        if not rows_t:
            per_type.append(
                {
                    "phish_type": tname,
                    "n": 0,
                    "accuracy": float("nan"),
                    "detection_rate (phishing)": float("nan"),
                    "false_positive_rate (legitimate)": float("nan"),
                }
            )
            continue

        correct = 0
        tp_t = fn_t = fp_t = tn_t = 0

        for row in rows_t:
            true_label = row["true"]
            pred_label = row["pred"]

            if pred_label == true_label:
                correct += 1

            if true_label == POS:
                if pred_label == POS:
                    tp_t += 1
                else:
                    fn_t += 1
            else:
                if pred_label == POS:
                    fp_t += 1
                else:
                    tn_t += 1

        acc_t = correct / len(rows_t)
        det_rate = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else float("nan")
        fp_rate = fp_t / (fp_t + tn_t) if (fp_t + tn_t) else float("nan")

        per_type.append(
            {
                "phish_type": tname,
                "n": len(rows_t),
                "accuracy": acc_t,
                "detection_rate (phishing)": det_rate,
                "false_positive_rate (legitimate)": fp_rate,
            }
        )

    per_type_df = pd.DataFrame(per_type)

    # Sort by amount of usable rows for that type
    if "n" in per_type_df.columns:
        per_type_df = per_type_df.sort_values(
            by=["n"],
            ascending=[False],
        )

    # Print results
    print(f"File: {args.csv}")
    print(f"Rows: {total_rows}")
    print(f"Missing predictions: {missing_pred_count} ({missing_pred_rate:.4f})")
    print(f"Coverage: {coverage:.4f}")
    print()

    print("Overall binary metrics - (positive = PHISHING), (negative = LEGITIMATE):")
    print(f"Usable rows (true+pred): {n}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1:        {f1:.4f}")
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