"""
Validates all critical logic of training.py WITHOUT needing to download any model.
Uses Phi-2 tokenizer (already cached) to simulate tokenization behavior.
"""
import json, re
from transformers import AutoTokenizer

# ---- mirror exact config from training.py ----
MODEL_NAME       = "microsoft/phi-2"   # use cached tokenizer as proxy
INPUT_CHAR_LIMIT = 1200
MAX_LENGTH       = 512
MIN_OUTPUT_LEN   = 20

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

PROMPT_TEMPLATE = (
    "### Instruction:\nSummarize the meeting.\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def clean_input(text):
    if "Meeting:\n" in text:
        text = text.split("Meeting:\n", 1)[-1]
    elif "Meeting: " in text:
        text = text.split("Meeting: ", 1)[-1]
    for p in ["Summarize the following meeting:", "Summarize the following meeting segment:", "Answer the query:"]:
        text = text.replace(p, "")
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_function(example):
    cleaned  = clean_input(example["input"])[:INPUT_CHAR_LIMIT]
    prompt   = PROMPT_TEMPLATE.format(input=cleaned)
    response = example["output"] + tokenizer.eos_token
    full     = prompt + response

    full_enc   = tokenizer(full,   truncation=True, max_length=MAX_LENGTH, padding=False)
    prompt_enc = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding=False)
    prompt_len = len(prompt_enc["input_ids"])

    labels = [-100] * len(full_enc["input_ids"])
    labels[prompt_len:] = full_enc["input_ids"][prompt_len:]

    if all(l == -100 for l in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    full_enc["labels"] = labels
    return full_enc

# ---- load real data ----
with open("data/processed/train.jsonl") as f:
    samples = [json.loads(l) for l in f][:500]

print(f"Testing {len(samples)} samples\n")

style1 = style2 = empty_resp = short_out = bad_clean = 0
for i, s in enumerate(samples):
    if len(s.get("output","")) < MIN_OUTPUT_LEN:
        short_out += 1
        continue
    if "Answer the query:" in s["input"]:
        style2 += 1
    else:
        style1 += 1

    out = tokenize_function(s)
    if len(out["input_ids"]) == 0:
        empty_resp += 1
        continue

    # Verify decoded response matches ground truth
    resp_ids = [t for t,l in zip(out["input_ids"], out["labels"]) if l != -100]
    resp_dec = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    gt       = s["output"].strip()

    # Check prefix is NOT in response
    if clean_input(s["input"])[:50] in resp_dec:
        bad_clean += 1
        print(f"[WARN] Sample {i}: response contains input! {resp_dec[:80]!r}")

print(f"Style 1 (Summarize): {style1}")
print(f"Style 2 (Query):     {style2}")
print(f"Short outputs filtered: {short_out}")
print(f"Truncation-killed samples: {empty_resp}")
print(f"Responses containing input (bad): {bad_clean}")
print(f"Valid samples: {style1+style2}")

# Visual verify 3 samples
print("\n===== VISUAL VERIFY =====")
good = [s for s in samples if len(s.get("output","")) >= MIN_OUTPUT_LEN]
for i in [0, 1, -1]:
    s = good[i]
    out = tokenize_function(s)
    if len(out["input_ids"]) == 0:
        print(f"Sample {i}: TRUNCATED")
        continue
    resp_ids = [t for t,l in zip(out["input_ids"], out["labels"]) if l != -100]
    resp_dec = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    total, masked = len(out["input_ids"]), sum(1 for l in out["labels"] if l == -100)
    print(f"\n--- Sample {i} ---")
    print(f"  Type: {'Query' if 'Answer the query:' in s['input'] else 'Summarize'}")
    print(f"  Tokens: total={total}, prompt={masked}, response={total-masked}")
    print(f"  GT     : {s['output'][:100]!r}")
    print(f"  Decoded: {resp_dec[:100]!r}")
    print(f"  Match: {resp_dec[:80] in s['output'] or s['output'][:80] in resp_dec}")

print("\nValidation complete.")
