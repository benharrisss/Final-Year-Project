import argparse
import pandas as pd
from metrics_utils import POS, NEG, clean_true_label, clean_pred_label, clean_type, compute_metrics

def calculate_brier(df):
    squared_error_sum = 0.0
    n_used = 0

    for _, row in df.iterrows():
        true_label = row["true"]
        if true_label is None:
            continue

        try:
            pred_prob = float(row["predicted_prob_phishing"])
        except Exception:
            continue
        
        if pd.isna(pred_prob):
            continue
        
        # Clamp predicted probability to [0, 1]
        if pred_prob < 0.0:
            pred_prob = 0.0
        elif pred_prob > 1.0:
            pred_prob = 1.0

        # Brier score is the squared error between the predicted probability and the true label
        true_binary = 1.0 if true_label == POS else 0.0
        squared_error = (pred_prob - true_binary) ** 2
        squared_error_sum += squared_error
        n_used += 1

    if n_used == 0:
        return float("nan"), 0
    
    return squared_error_sum / n_used, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to SecBERT results CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Needs to include true and predicted labels
    required = ["true_label", "predicted_label"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"CSV must contain column: {col}")

    has_type = "phish_type" in df.columns
    has_prob = "predicted_prob_phishing" in df.columns

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

    # Only evaluate rows where we have both true and pred
    usable_rows = []
    for _, row in df.iterrows():
        if row["true"] is None:
            continue
        if row["pred"] is None:
            continue
        usable_rows.append({"true": row["true"], "pred": row["pred"], "phish_type": row["phish_type"]})

    overall = compute_metrics(usable_rows)

    # Calculate Brier score if they exist
    brier = None
    brier_used = 0
    if has_prob:
        brier, brier_used = calculate_brier(df)

    # Per phish_type stats
    per_type = []
    types = sorted(set(df["phish_type"]))

    for tname in types:
        # Use all usable rows that contain true, pred and a phish_type that matches the list of types
        rows_t = [r for r in usable_rows if r["phish_type"] == tname]
        m = compute_metrics(rows_t)

        # detection_rate_if_phish = TP / (TP+FN) within that type 
        # false_positive_rate_if_legit = FP / (FP+TN) within that type
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
    if brier is not None:
        print(f"Brier score (lower is better, 0.0 is perfect): {brier:.4f} (n={brier_used})")
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