import pandas as pd
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import time

DEFAULT_MODEL_PATH = "./models/deepseek-llm-7b-chat"
DEFAULT_INPUT_CSV = "final_phish+legit_2200.csv"
DEFAULT_OUTPUT_CSV = "results.csv"
MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 4096

tokenizer = None
model = None

def truncate_body(text, n_tokens):
    if tokenizer is None:
        return text
    if n_tokens <= 0:
        return ""
    ids = tokenizer(text,
        add_special_tokens=False, 
        truncation=True, 
        max_length=n_tokens, 
        return_attention_mask=False)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

def build_prompt(subject, body):
    # Just specify the required JSON keys/values in plain text.
    body = truncate_body(body, MAX_INPUT_TOKENS)
    return f"""
Output rules:
- Output ONLY a single valid JSON object
- DO NOT output markdown, code fences, or extra text

Return ONLY a valid JSON object with exactly these keys:
- classification: "PHISHING" or "LEGITIMATE"
- reasoning: a short explanation

Classify the following email as either PHISHING or LEGITIMATE.

Email subject: {subject}

Email body:
{body}
""".strip()

def extract_json(text):
    # Parse the last valid JSON object in text
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "classification" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None

def normalise_weird_text(t):
    # e.g. Turns "PHISH!ING" into simply "phishing"
    return re.sub(r"[^a-z]", "", (t or "").lower())

def recover_classification(text):
    t = text.lower()
    idx = t.find("classification")
    if idx == -1:
        return None

    # small local region after the key
    local_region = text[idx : idx + 50]
    norm = normalise_weird_text(local_region)

    if "phishing" in norm:
        return "PHISHING"
    if "legitimate" in norm:
        return "LEGITIMATE"
    return None

def fallback_classification(text):
    t = text.lower()
    if "phishing" in t:
        return "PHISHING"
    if "legitimate" in t:
        return "LEGITIMATE"
    return None

def format_for_model(prompt):
    # If the tokenizer has a chat template then use it.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def sanitise_output(text):
    if not text:
        return text
    t = text.replace("!", "")
    t = t.replace("```json", "").replace("```", "")
    return t.strip()

@torch.inference_mode()
def classify_email(subject, body):
    prompt = build_prompt(subject, body)
    formatted = format_for_model(prompt)

    inputs = tokenizer(formatted, 
        return_tensors="pt",
        truncation=True, 
        max_length=MAX_INPUT_TOKENS, 
        add_special_tokens=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Sync before and after for more accurate timing 
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

    # Decode ONLY the newly generated tokens
    gen_ids = out[0][inputs["input_ids"].shape[1]:]

    output_tokens = int(gen_ids.numel())
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Hard-trim anything after the final JSON brace, but only for JSON parsing
    completion_for_parsing = completion
    last_brace = completion_for_parsing.rfind("}")
    if last_brace != -1:
        completion_for_parsing = completion_for_parsing[: last_brace + 1].strip()

    parsed = extract_json(completion_for_parsing)
    if parsed:
        label = str(parsed.get("classification", "")).strip().upper()
        if label in ("PHISHING", "LEGITIMATE"):
            return label, parsed.get("reasoning"), None, completion, output_tokens, generation_seconds

    sanitised = sanitise_output(completion)
    parsed2 = extract_json(sanitised)
    if parsed2:
        label = str(parsed2.get("classification", "")).strip().upper()
        if label in ("PHISHING", "LEGITIMATE"):
            return label, parsed2.get("reasoning"), "JSON_PARSE_FAILED_BUT_SANITISED", completion, output_tokens, generation_seconds

    recovered_label = recover_classification(completion_for_parsing)
    if recovered_label in ("PHISHING", "LEGITIMATE"):
        return recovered_label, None, "JSON_PARSE_FAILED_BUT_RECOVERED", completion, output_tokens, generation_seconds

    fallback_label = fallback_classification(completion)
    return fallback_label, None, "JSON_PARSE_FAILED", completion, output_tokens, generation_seconds

def main():
    global tokenizer, model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the local model directory.")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to the input CSV file.")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save the output CSV file.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models/tokenizers that require custom code in the model repo.")
    args = parser.parse_args()

    # Overall Program time (includes load + CSV + generation)
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

    print("device map:", getattr(model, "hf_device_map", None))

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

        predicted_label, reasoning, error, raw_response, output_tokens, generation_seconds = classify_email(subject, body)

        total_output_tokens += output_tokens
        total_generation_seconds += generation_seconds

        results.append({"email_id": i,
            "subject": subject,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "phish_type": phish_type,
            "reasoning": reasoning,
            "error": error,
            "raw_response": raw_response})

        print(f"[{i+1}/{len(df)}] Done: Predicted={predicted_label}")

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("Results saved to:", args.output_csv)

    total_seconds = time.perf_counter() - total_t0
    mean_output_tokens_per_sec = (total_output_tokens / total_generation_seconds)
    mean_generation_seconds_per_email = (total_generation_seconds / len(df))

    print("\nTiming summary: ")
    print(f"Model load time (s): {load_seconds:.2f}")
    print(f"Total Test time (s): {total_seconds:.2f}")
    print(f"Total generation time (s): {total_generation_seconds:.2f}")
    print(f"Mean generation time per email (s): {mean_generation_seconds_per_email:.4f}")
    print(f"Mean output tokens/sec: {mean_output_tokens_per_sec:.2f}")

if __name__ == "__main__":
    main()
