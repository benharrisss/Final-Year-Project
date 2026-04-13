import argparse
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_STR = {0: "LEGITIMATE", 1: "PHISHING"}


class EmailDataset(Dataset):
    # Dataset wrapper around the CSV, to be used with DataLoader for batching
    def __init__(self, df):
        if "label" not in df.columns:
            raise SystemExit("CSV must contain 'label' column with 0=legit, 1=phishing.")
        self.df = df.reset_index(drop=True).copy()
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject = str(row["subject"])
        body = str(row["body"])
        # Concatenate subject and body for input, with a separator
        text = (subject.strip() + "\n\n" + body.strip()).strip()
        label = int(row["label"])
        return text, label


def make_collate_fn(tokenizer, max_tokens):
    # DataLoader's collate_fn to tokenize a batch of texts and prepare tensors for the model
    def collate(batch):
        # Labels needed for training
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        )
        # Huggingface's sequence classification models expect labels in the batch for training
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding
    return collate


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    # Compute accuracy on the given DataLoader to monitor training progress
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()

    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="SecBERT", help="Local path to the model")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--training-rounds", dest="training_rounds", type=int, default=3, help="Number of full passes over the training set")
    ap.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=44)

    args = ap.parse_args()

    # Time the entire run (model load + training + prediction)
    total_t0 = time.perf_counter()

    # Sets torch random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    print("Loading model/tokenizer...")
    load_t0 = time.perf_counter()

    # Tokenizer converts raw text into tokens for the model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Model is a classification head on top of a transformer encoder
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_t0

    # Each dataframe row is turned into a single string + label
    train_ds = EmailDataset(train_df)
    test_ds = EmailDataset(test_df)

    # Collate function starts batch tokenization process
    collate_fn = make_collate_fn(tokenizer, max_tokens=args.max_tokens)

    # During training, the dataset is shuffled so ordering isnt learned by the model
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # During testing, shuffling is not necessary as only one pass
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # AdamW is the standard optimizer for transformer fine-tuning
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_t0 = time.perf_counter()
    for round_idx in range(1, args.training_rounds + 1):
        model.train()
        for batch in train_loader:
            # Move batch to the same device as the model (GPU or CPU)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass, compute loss, backward pass, and then update model weights
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        # After each training round, evaluate accuracy on the test set to monitor progress
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Training round {round_idx}/{args.training_rounds} | test accuracy={acc:.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_t0

    # Run the fine-tuned model on the test set
    model.eval()
    rows = []

    pred_t0 = time.perf_counter()
    for i in range(len(test_ds)):
        text, true_label = test_ds[i]
        row = test_df.iloc[i]

        subject = str(row["subject"])
        phish_type = str(row["phish_type"])

        # Tokenize the text and prepare tensors for the model - max_length prevents context overflow
        encoding = tokenizer(text, truncation=True, max_length=args.max_tokens, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Forward pass but without labels for prediction
        logits = model(**encoding).logits[0]

        # Convert logits to probabilities and predicted classification (0 or 1)
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

    # Timing measurements printed to terminal and saved in logs
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