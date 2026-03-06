import argparse
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_STR = {0: "LEGITIMATE", 1: "PHISHING"}


class EmailDataset(Dataset):
    """
    Turns a CSV DataFrame into (text, label) items.
    Text = subject + "\\n\\n" + body.
    """
    def __init__(self, df: pd.DataFrame):
        if "label" not in df.columns:
            raise SystemExit("CSV must contain 'label' column with 0=legit, 1=phishing.")
        self.df = df.reset_index(drop=True).copy()
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", "") or "")
        text = (subject.strip() + "\n\n" + body.strip()).strip()
        label = int(row["label"])
        return text, label


def make_collate_fn(tokenizer, max_tokens: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


def evaluate_accuracy(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()

    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="jackaduma/SecBERT", help="HuggingFace model name or local path")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--training-rounds", dest="training_rounds", type=int, default=3, help="Number of full passes over the training set")
    ap.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=44)

    args = ap.parse_args()

    total_t0 = time.perf_counter()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    load_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_t0

    train_ds = EmailDataset(train_df)
    test_ds = EmailDataset(test_df)

    collate_fn = make_collate_fn(tokenizer, max_tokens=args.max_tokens)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_t0 = time.perf_counter()
    for round_idx in range(1, args.training_rounds + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Training round {round_idx}/{args.training_rounds} | test accuracy={acc:.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_t0

    model.eval()
    rows = []

    pred_t0 = time.perf_counter()
    for i in range(len(test_ds)):
        text, true_label = test_ds[i]

        subject = str(test_df.iloc[i].get("subject", "") or "")
        phish_type = str(test_df.iloc[i].get("phish_type", "") or "")

        enc = tokenizer(text, truncation=True, max_length=args.max_tokens, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits[0]
        prob_phish = torch.softmax(logits, dim=-1)[1].item()
        pred_id = int(torch.argmax(logits).item())

        rows.append(
            {
                "email_id": i,
                "subject": subject,
                "phish_type": phish_type,
                "true_label": int(true_label),
                "predicted_label": LABEL_STR[pred_id],
                "predicted_prob_phishing": float(prob_phish),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pred_seconds = time.perf_counter() - pred_t0

    total_seconds = time.perf_counter() - total_t0

    print("\nTiming summary:")
    print(f"Model load time (s): {load_seconds:.2f}")
    print(f"Training time (s): {train_seconds:.2f}")
    print(f"Prediction time (s): {pred_seconds:.2f}")
    print(f"Total program time (s): {total_seconds:.2f}")

    n_total = len(train_ds) + len(test_ds)
    if total_seconds > 0:
        print(f"Examples/sec (overall): {n_total / total_seconds:.2f}")
    if pred_seconds > 0:
        print(f"Examples/sec (prediction only): {len(test_ds) / pred_seconds:.2f}")

    print("Wrote:", args.output_csv)


if __name__ == "__main__":
    main()