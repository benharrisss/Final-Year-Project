import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--train-out-csv", required=True)
    ap.add_argument("--test-out-csv", required=True)
    ap.add_argument("--train-phish", type=int, default=50)
    ap.add_argument("--train-legit", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    if "label" not in df.columns:
        raise SystemExit("Input CSV must contain a 'label' column with 0=legit, 1=phishing.")

    # Ensure the label is numeric (0 or 1 expected)
    df["label"] = df["label"].astype(int)

    phish_df = df[df["label"] == 1]
    legit_df = df[df["label"] == 0]

    # Fail quick if there aren't enough rows to sample the requested number for training
    if len(phish_df) < args.train_phish:
        raise SystemExit(f"Not enough phishing rows to sample {args.train_phish}. Found {len(phish_df)}.")
    if len(legit_df) < args.train_legit:
        raise SystemExit(f"Not enough legit rows to sample {args.train_legit}. Found {len(legit_df)}.")

    # Fixed random state ensures a reproducible split
    train_phish = phish_df.sample(n=args.train_phish, random_state=44)
    train_legit = legit_df.sample(n=args.train_legit, random_state=44)

    # Shuffle the training partition to avoid ordering bias
    train_df = pd.concat([train_phish, train_legit]).sample(frac=1.0, random_state=44)
    
    # Test set defined as the remaining rows to prevent data leakage
    test_df = df.drop(index=train_df.index)

    train_df.reset_index(drop=True).to_csv(args.train_out_csv, index=False)
    test_df.reset_index(drop=True).to_csv(args.test_out_csv, index=False)

    print(f"Wrote train: {args.train_out_csv} ({len(train_df)} rows)")
    print(f"Wrote test: {args.test_out_csv} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()