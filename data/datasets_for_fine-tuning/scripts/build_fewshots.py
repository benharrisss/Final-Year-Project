import argparse
import json
import pandas as pd


def label_to_str(v):
    return "PHISHING" if int(v) == 1 else "LEGITIMATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summaries-csv", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--num-phish", type=int, default=5)
    ap.add_argument("--num-legit", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.train_summaries_csv)

    for c in ["subject", "summary"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    label_col = "true_label"
    if label_col not in df.columns:
        raise SystemExit(f"Missing required column: {label_col}")

    # Filter out empty/blank summaries to avoid training on uninformative examples
    mask = df["summary"].notna() & (df["summary"].astype(str).str.strip() != "")
    df = df[mask].copy()
    
    df[label_col] = df[label_col].astype(int)

    # Fixed random state ensures reproducible few-shot selection
    phish = df[df[label_col] == 1].sample(n=args.num_phish, random_state=444)
    legit = df[df[label_col] == 0].sample(n=args.num_legit, random_state=444)

    shots_df = pd.concat([phish, legit]).sample(frac=1.0, random_state=444)

    # Write out the few-shots in JSONL format expected by the fine-tuning script
    with open(args.train_jsonl, "w", encoding="utf-8") as f:
        for _, row in shots_df.iterrows():
            obj = {
                "subject": str(row["subject"]),
                "summary": str(row["summary"]),
                "label": label_to_str(row[label_col]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(shots_df)} fixed shots to {args.train_jsonl}")


if __name__ == "__main__":
    main()