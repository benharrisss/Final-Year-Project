import argparse
import time
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "./models/gemma-2-9-it"
DEFAULT_INPUT_CSV = "final_phish+legit_2200.csv"
DEFAULT_OUTPUT_CSV = "results.csv"

MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 3076
STRATEGY_CHOICES = ["rephrasing", "context_manipulation", "logical_contradiction"]

# Loaded once in main() and used by the helper functions below
tokenizer = None
model = None


def truncate_body(text, n_tokens):
    if tokenizer is None:
        return text
    if n_tokens <= 0:
        return ""
    # Truncate the body to n_tokens, stops going over context window
    ids = tokenizer(text,
        add_special_tokens=False, 
        truncation=True, 
        max_length=n_tokens, 
        return_attention_mask=False)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_rephrasing_prompt(subject, body):
    body = truncate_body(body, MAX_INPUT_TOKENS)
    # Just specify the required JSON keys/values in plain text
    return f"""
Rephrase the following email.
Rephrasing rules:
- Preserve the original meaning of the email.
- Rearrange sentence structure, change wording, and modify structure to create a new version.
- Rearrange paragraph structure, but do not add or remove paragraphs.

Note: Just return the modified body, the subject will be preserved as is.

Email subject: {subject}

Email body:
{body}
""".strip()


def build_context_manipulation_prompt(subject, body):
    body = truncate_body(body, MAX_INPUT_TOKENS)
    # Just specify the required JSON keys/values in plain text
    return f"""
Apply context manipulation to the following email.
Context manipulation rules:
- Preserve the original meaning of the email.
- Add misleading narrative context to obscure malicious intent of the email.

Note: Just return the modified body, the subject will be preserved as is.

Email subject: {subject}

Email body:
{body}
""".strip()


def build_logical_contradiction_prompt(subject, body):
    body = truncate_body(body, MAX_INPUT_TOKENS)
    # Just specify the required JSON keys/values in plain text
    return f"""
Apply logical contradiction to the following email.
Logical contradiction rules:
- Preserve the original meaning of the email.
- Embed inconsistent or confusing statements within the email to create logical contradictions.

Note: Just return the modified body, the subject will be preserved as is.

Email subject: {subject}

Email body:
{body}
""".strip()


def format_for_model(prompt):
    # If the tokenizer has a chat template then use it
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        # Some models require structured format to behave correctly
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.inference_mode()
def classify_email(subject, body, strategy):
    if strategy == "rephrasing":
        prompt = build_rephrasing_prompt(subject, body)
    elif strategy == "context_manipulation":
        prompt = build_context_manipulation_prompt(subject, body)
    elif strategy == "logical_contradiction":
        prompt = build_logical_contradiction_prompt(subject, body)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    formatted = format_for_model(prompt)

    inputs = tokenizer(formatted, 
        return_tensors="pt",
        truncation=True, 
        max_length=MAX_INPUT_TOKENS, 
        add_special_tokens=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Sync before and after for more accurate GPU timing 
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model.generate(**inputs,
        max_new_tokens=MAX_NEW_TOKENS, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - t0

    # Decode ONLY the newly generated tokens (not the prompt)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    output_tokens = int(gen_ids.numel())
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if not completion:
        return None, output_tokens, generation_seconds
    return completion, output_tokens, generation_seconds


def main():
    global tokenizer, model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the local model directory.")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to the input CSV file.")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save the output CSV file.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models/tokenizers that require custom code in the model repo.")
    parser.add_argument("--strategy", type=str, choices=STRATEGY_CHOICES, required=True, help="Email manipulation strategy to apply before classification.")
    args = parser.parse_args()

    # Overall Program time (includes load + CSV read + generation)
    total_t0 = time.perf_counter()

    print("Loading model...")
    load_t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
        local_files_only=True, 
        trust_remote_code=args.trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=args.trust_remote_code)

    load_seconds = time.perf_counter() - load_t0

    # After loading the model, read the CSV and run classification on each row
    df = pd.read_csv(args.input_csv)
    results = []

    print(f"Processing {len(df)} emails...\n")

    total_output_tokens = 0
    total_generation_seconds = 0.0

    for i, row in df.iterrows():
        subject = str(row["subject"])
        body = str(row["body"])
        true_label = row["label"]
        phish_type = str(row["phish_type"])

        response, output_tokens, generation_seconds = classify_email(subject, body, strategy=args.strategy)

        total_output_tokens += output_tokens
        total_generation_seconds += generation_seconds

        # Store the results for each classification - will be converted to a DataFrame
        results.append({"subject": subject,
            "body": response,
            "label": true_label,
            "phish_type": phish_type,})

        print(f"[{i+1}/{len(df)}] Done")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("Results saved to:", args.output_csv)

    total_seconds = time.perf_counter() - total_t0
    mean_output_tokens_per_sec = (total_output_tokens / total_generation_seconds)
    mean_generation_seconds_per_email = (total_generation_seconds / len(df))

    # Timing measurements printed to terminal and saved in logs
    print("\nTiming summary: ")
    print(f"Model load time (s): {load_seconds:.2f}")
    print(f"Total Test time (s): {total_seconds:.2f}")
    print(f"Total generation time (s): {total_generation_seconds:.2f}")
    print(f"Mean generation time per email (s): {mean_generation_seconds_per_email:.4f}")
    print(f"Mean output tokens/sec: {mean_output_tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()