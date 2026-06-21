import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--phish", type=int, default=990)
    ap.add_argument("--legit", type=int, default=110)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    if "label" not in df.columns:
        raise SystemExit("Input CSV must contain a 'label' column with 0=legit, 1=phishing.")

    # Ensure the label is numeric (0 or 1 expected)
    df["label"] = df["label"].astype(int)

    phish_df = df[df["label"] == 1]
    legit_df = df[df["label"] == 0]

    # Fail quick if there aren't enough rows to sample the requested number for training
    if len(phish_df) < args.phish:
        raise SystemExit(f"Not enough phishing rows to sample {args.phish}. Found {len(phish_df)}.")
    if len(legit_df) < args.legit:
        raise SystemExit(f"Not enough legit rows to sample {args.legit}. Found {len(legit_df)}.")

    # Fixed random state ensures a reproducible split
    # Shuffle the training partition to avoid ordering bias
    out_df = pd.concat([
        phish_df.sample(n=args.phish, random_state=44),
        legit_df.sample(n=args.legit, random_state=44),
    ]).sample(frac=1.0, random_state=44)

    # Output the resulting dataset to a CSV file
    out_df.reset_index(drop=True).to_csv(args.out_csv, index=False)

    print(f"Wrote output: {args.out_csv} ({len(out_df)} rows)")
    print(f"Phishing: {args.phish}, Legit: {args.legit}")


if __name__ == "__main__":
    main()