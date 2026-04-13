import argparse
import time
import json
import re
import random
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "./models/deepseek-llm-7b-chat"
DEFAULT_INPUT_CSV = "final_phish+legit_2200.csv"
DEFAULT_OUTPUT_CSV = "results_summary_then_fewshot_classify.csv"

MAX_INPUT_TOKENS = 3076
SUMMARY_MAX_NEW_TOKENS = 256
CLASSIFY_MAX_NEW_TOKENS = 256

RAND_SEED = 44

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
        return_attention_mask=False,)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_summary_prompt(subject, body):
    body = truncate_body(body, MAX_INPUT_TOKENS)
    # Just specify the required JSON keys/values in plain text
    return f"""
Output rules:
- Output ONLY a single valid JSON object
- DO NOT output markdown, code fences, or extra text

Return ONLY a valid JSON object with exactly these keys:
- summary: string

Summarise the email content concisely. Preserve:
- the sender identity and why they are contacting the recipient
- what is being requested or asked of the recipient
- any urgency/threats/deadlines
- any links, attachments, phone numbers, potential sender claims
- what sensitive information is being requested (if any)

Email subject: {subject}

Email body:
{body}
""".strip()


def build_fewshot_block(shots):
    if not shots:
        return ""
    # Build fewshot examples block in recognisable format for model
    parts = ["Examples:"]
    for i, ex in enumerate(shots, start=1):
        parts.append(
            f"""
Example {i}
Email subject: {ex["subject"]}
Email summary: {ex["summary"]}
Output: {{"classification": "{ex["label"]}"}}
""".strip()
        )
    return "\n\n".join(parts).strip() + "\n\n"


def build_classify_prompt(subject, summary, shots):
    fewshot = build_fewshot_block(shots)
    # Just specify the required JSON keys/values in plain text
    # Use summary and fewshot examples to help classify
    return f"""
Output rules:
- Output ONLY a single valid JSON object
- DO NOT output markdown, code fences, or extra text

Return ONLY a valid JSON object with exactly these keys:
- classification: "PHISHING" or "LEGITIMATE"
- reasoning: a short explanation

Use the following examples to help classify the unseen email.

{fewshot}

Classify the following unseen email as either PHISHING or LEGITIMATE.
Do this using the sumarised email content and subject.

Email subject: {subject}

Email summary:
{summary}
""".strip()


def format_for_model(prompt):
    # If the tokenizer has a chat template then use it
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        # Some models require structured format to behave correctly
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.inference_mode()
def generate_text(prompt, max_new_tokens):
    formatted = format_for_model(prompt)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Sync before and after for more accurate GPU timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - t0

    # Decode ONLY the newly generated tokens (not the prompt)
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    output_tokens = int(gen_ids.numel())
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return completion, output_tokens, generation_seconds


def extract_json(text):
    # Parse the last valid JSON object in text (last object typically final)
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def sanitise_output(text):
    if not text:
        return text
    # Remove common formatting issues but not "!" (Qwen2.5-7B)
    return text.replace("```json", "").replace("```", "").strip()


def normalise_weird_text(t):
    # e.g. Turns "PHISH!ING" into simply "phishing"
    return re.sub(r"[^a-z]", "", (t or "").lower())


def recover_classification(text):
    # Find "classification" and search near it for label
    t = text.lower()
    idx = t.find("classification")
    if idx == -1:
        return None

    # Small local region after the key
    local_region = text[idx : idx + 50]
    norm = normalise_weird_text(local_region)

    if "phishing" in norm:
        return "PHISHING"
    if "legitimate" in norm:
        return "LEGITIMATE"
    return None


def fallback_classification(text):
    # As a last effort, search the whole text for phishing/legitimate keywords
    t = (text or "").lower()
    if "phishing" in t:
        return "PHISHING"
    if "legitimate" in t:
        return "LEGITIMATE"
    return None


def summarise_email(subject, body, max_new_tokens):
    prompt = build_summary_prompt(subject, body)
    completion, output_tokens, generation_seconds = generate_text(prompt, max_new_tokens=max_new_tokens)

    # Hard-trim anything after the final JSON brace - remove excess output
    completion_for_parsing = completion
    last_brace = completion_for_parsing.rfind("}")
    if last_brace != -1:
        completion_for_parsing = completion_for_parsing[: last_brace + 1].strip()

    # Use any valid JSON found in output as summary
    parsed = extract_json(completion_for_parsing) or extract_json(sanitise_output(completion))
    if parsed and isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
        return parsed["summary"].strip(), None, completion, output_tokens, generation_seconds

    # If parsing failed, return summary generation failed + raw response
    if completion.strip():
        return completion.strip(), "SUMMARY_JSON_PARSE_FAILED", completion, output_tokens, generation_seconds

    return None, "SUMMARY_FAILED_EMPTY", completion, output_tokens, generation_seconds


def classify_from_summary(subject, summary, shots, max_new_tokens):
    prompt = build_classify_prompt(subject, summary, shots)
    completion, output_tokens, generation_seconds = generate_text(prompt, max_new_tokens=max_new_tokens)

    # Hard-trim anything after the final JSON brace - remove excess output
    completion_for_parsing = completion
    last_brace = completion_for_parsing.rfind("}")
    if last_brace != -1:
        completion_for_parsing = completion_for_parsing[: last_brace + 1].strip()

    parsed = extract_json(completion_for_parsing)
    if parsed:
        label = str(parsed.get("classification", "")).strip().upper()
        if label in ("PHISHING", "LEGITIMATE"):
            return label, parsed.get("reasoning"), None, completion, output_tokens, generation_seconds

    # If parsing failed, attempt to sanitise the output and parse again
    sanitised = sanitise_output(completion)
    parsed2 = extract_json(sanitised)
    if parsed2:
        label = str(parsed2.get("classification", "")).strip().upper()
        if label in ("PHISHING", "LEGITIMATE"):
            return label, parsed2.get("reasoning"), "JSON_PARSE_FAILED_BUT_SANITISED", completion, output_tokens, generation_seconds

    # If it still fails, attempt to recover the classification label with some heuristics
    recovered_label = recover_classification(completion_for_parsing)
    if recovered_label in ("PHISHING", "LEGITIMATE"):
        return recovered_label, None, "JSON_PARSE_FAILED_BUT_RECOVERED", completion, output_tokens, generation_seconds

    # As a last effort, use any fallback label found, and return raw response
    fallback_label = fallback_classification(completion)
    return fallback_label, None, "JSON_PARSE_FAILED", completion, output_tokens, generation_seconds


def load_shots_jsonl(path):
    shots: List[dict] = []
    # Load fewshot examples from JSONL file - validating required fields
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            subject = str(obj.get("subject", "")).strip()
            summary = str(obj.get("summary", "")).strip()
            label = str(obj.get("label", "")).strip().upper()

            if not subject or not summary:
                continue
            if label not in ("PHISHING", "LEGITIMATE"):
                continue

            shots.append({"subject": subject, "summary": summary, "label": label})
    return shots


def pick_phish_legit_shots(phish_shots, legit_shots, num_phish, num_legit, rng):
    # Ensure there are enough shots as specified to sample from
    if len(phish_shots) < num_phish:
        raise SystemExit(f"Not enough phishing shots to sample {num_phish}. Found {len(phish_shots)}.")
    if len(legit_shots) < num_legit:
        raise SystemExit(f"Not enough legitimate shots to sample {num_legit}. Found {len(legit_shots)}.")

    # Sample randomly from provided shots as to not cherry-pick shots
    picked_phish = rng.sample(phish_shots, num_phish)
    picked_legit = rng.sample(legit_shots, num_legit)

    shots = picked_phish + picked_legit
    rng.shuffle(shots)
    return shots


def main():
    global tokenizer, model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--shots-jsonl", type=str, required=True)
    parser.add_argument("--shots-mode", type=str, choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument("--num-phish", type=int, default=5)
    parser.add_argument("--num-legit", type=int, default=5)
    args = parser.parse_args()

    # Overall Program time (includes load + CSV read + generation)
    total_t0 = time.perf_counter()

    print("Loading model...")
    load_t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )

    load_seconds = time.perf_counter() - load_t0

    shots = load_shots_jsonl(args.shots_jsonl)
    if not shots:
        raise SystemExit("No valid shots loaded. Check shots JSONL content.")

    # Make sure all shots have labels to make sure they are effective for model
    phish_shots = [s for s in shots if s["label"] == "PHISHING"]
    legit_shots = [s for s in shots if s["label"] == "LEGITIMATE"]

    print(f"Loaded {len(shots)} shots from {args.shots_jsonl}. phishing={len(phish_shots)}, legitimate={len(legit_shots)}")

    # Fixed fewshot selection (same shots for every email)
    fixed_shots = None
    if args.shots_mode == "fixed":
        rng = random.Random(RAND_SEED)
        fixed_shots = pick_phish_legit_shots(phish_shots, legit_shots, args.num_phish, args.num_legit, rng)
        print(f"Using fixed shots: {len(fixed_shots)} total, with {args.num_phish} phishing and {args.num_legit} legitimate.")

    # After loading the model, read CSV & run summary+fewshot+classify on each row
    df = pd.read_csv(args.input_csv)
    results = []

    print(f"Processing {len(df)} emails...\n")

    total_summary_tokens = 0
    total_summary_seconds = 0.0
    total_classify_tokens = 0
    total_classify_seconds = 0.0

    for i, row in df.iterrows():
        subject = str(row["subject"])
        body = str(row["body"])
        true_label = row["label"]
        phish_type = str(row["phish_type"])

        summary, sum_error, sum_raw, sum_tokens, sum_seconds = summarise_email(
            subject, body, max_new_tokens=SUMMARY_MAX_NEW_TOKENS
        )
        total_summary_tokens += sum_tokens
        total_summary_seconds += sum_seconds

        # If summary successful, attempt classification
        if summary:
            if args.shots_mode == "fixed":
                shots = fixed_shots
            else:
                # Dynamic fewshot selection per email decided at runtime
                rng = random.Random(RAND_SEED + int(i))
                shots = pick_phish_legit_shots(phish_shots, legit_shots, args.num_phish, args.num_legit, rng)

            predicted_label, reasoning, cls_error, cls_raw, cls_tokens, cls_seconds = classify_from_summary(
                subject, summary, shots=shots, max_new_tokens=CLASSIFY_MAX_NEW_TOKENS
            )
        else:
            predicted_label, reasoning, cls_error, cls_raw, cls_tokens, cls_seconds = (
                None, None, "NO_SUMMARY_AND_NO_CLASSIFICATION", "", 0, 0.0
            )

        total_classify_tokens += cls_tokens
        total_classify_seconds += cls_seconds

        # Store the results for each classification - will be converted to a DataFrame
        results.append(
            {
                "email_id": i,
                "subject": subject,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "phish_type": phish_type,
                "summary": summary,
                "summary_error": sum_error,
                "reasoning": reasoning,
                "classify_error": cls_error,
                "summary_raw_response": sum_raw,
                "classify_raw_response": cls_raw,
            }
        )

        print(f"[{i+1}/{len(df)}] Done: Predicted={predicted_label}")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("Results saved to:", args.output_csv)

    total_seconds = time.perf_counter() - total_t0
    summary_tok_per_sec = total_summary_tokens / total_summary_seconds
    classify_tok_per_sec = total_classify_tokens / total_classify_seconds
    mean_generation_seconds_per_email = total_classify_seconds / len(df)

    # Timing measurements printed to terminal and saved in logs
    print("\nTiming summary:")
    print(f"Model load time (s): {load_seconds:.2f}")
    print(f"Total program time (s): {total_seconds:.2f}")
    print(f"Summary stage time (s): {total_summary_seconds:.2f} | tokens/s: {summary_tok_per_sec:.2f}")
    print(f"Classify stage time (s): {total_classify_seconds:.2f} | tokens/s: {classify_tok_per_sec:.2f}")
    print(f"Mean generation time per email (s): {mean_generation_seconds_per_email:.2f}")


if __name__ == "__main__":
    main()