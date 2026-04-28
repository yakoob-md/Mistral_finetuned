"""
clean_data.py — Dataset cleaning script for Kaggle
===================================================
Run this BEFORE training.py.
Produces: /kaggle/working/clean_train.jsonl
          /kaggle/working/clean_val.jsonl

Input paths (from your existing Kaggle dataset):
  /kaggle/input/datasets/yakoob2345/processed/train.jsonl
  /kaggle/input/datasets/yakoob2345/processed/val.jsonl
"""

import json, re, os
from pathlib import Path

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
KAGGLE_DIR = "/kaggle/input/datasets/yakoob2345/processed"
LOCAL_DIR  = "./data/processed"
IN_DIR     = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else LOCAL_DIR

TRAIN_IN  = f"{IN_DIR}/train.jsonl"
VAL_IN    = f"{IN_DIR}/val.jsonl"

OUT_DIR   = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
TRAIN_OUT = f"{OUT_DIR}/clean_train.jsonl"
VAL_OUT   = f"{OUT_DIR}/clean_val.jsonl"

MIN_OUTPUT_WORDS  = 40    # drop summaries with fewer words
MAX_INPUT_CHARS   = 1000  # smart-truncate inputs longer than this

# ─────────────────────────────────────────
# INPUT CLEANING
# ─────────────────────────────────────────
# Filler words to strip from transcripts
FILLERS = re.compile(
    r'\b(uh+|um+|hmm+|mm+|mhm|erm|ah+|eh+|oh+|yeah|yep|nope|okay|ok)\b',
    re.IGNORECASE
)

def clean_input(text: str) -> str:
    # 1. Strip dataset-style prefixes
    if "Meeting:\n" in text:
        text = text.split("Meeting:\n", 1)[-1]
    elif "Meeting: " in text:
        text = text.split("Meeting: ", 1)[-1]
    for prefix in [
        "Summarize the following meeting:",
        "Summarize the following meeting segment:",
        "Answer the query:",
    ]:
        text = text.replace(prefix, "")

    # 2. Remove noise tokens: {disfmarker}, {vocalsound}, {gap}, {breath}, etc.
    text = re.sub(r'\{[^}]+\}', ' ', text)

    # 3. Remove speaker tags: "Grad F:", "PhD C:", "MIO072:", "PM:", "User Interface:" etc.
    text = re.sub(r'\b[A-Z][a-zA-Z\s]{0,20}:\s*', '', text)   # "Project Manager: "
    text = re.sub(r'\b[A-Z]{1,6}\d{0,4}\s*:\s*', '', text)    # "MIO072: ", "PM: "

    # 4. Remove filler words
    text = FILLERS.sub(' ', text)

    # 5. Remove repeated consecutive words (stutters: "the the the")
    text = re.sub(r'\b(\w+)( \1)+\b', r'\1', text, flags=re.IGNORECASE)

    # 6. Remove isolated single letters (e.g. "I I" artifacts)
    text = re.sub(r'(?<!\w)\b[bcdefghjklmnopqrstuvwxyz]\b(?!\w)', '', text, flags=re.IGNORECASE)

    # 7. Collapse whitespace + punctuation artifacts
    text = re.sub(r'\s*[,;]\s*[,;]+', ',', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def smart_truncate(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Keep beginning + end — conclusions are usually at the end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n[...]\n" + text[-half:].lstrip()

# ─────────────────────────────────────────
# OUTPUT CLEANING
# ─────────────────────────────────────────
def clean_output(text: str) -> str:
    # Strip any speaker tags that leaked into the summary
    text = re.sub(r'\b[A-Z][a-zA-Z\s]{0,20}:\s*', '', text)
    text = re.sub(r'\b[A-Z]{1,6}\d{0,4}\s*:\s*', '', text)
    # Remove noise tokens
    text = re.sub(r'\{[^}]+\}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ─────────────────────────────────────────
# FILTERING RULES
# ─────────────────────────────────────────
def is_valid(inp_clean: str, out_clean: str) -> tuple[bool, str]:
    """Returns (keep, reason_if_dropped)."""
    # 1. Output too short
    if len(out_clean.split()) < MIN_OUTPUT_WORDS:
        return False, f"output too short ({len(out_clean.split())} words)"

    # 2. Output still contains speaker-tagged lines (transcript leaked into label)
    speaker_hits = len(re.findall(r'\b[A-Z]{2,6}\d*\s*:', out_clean))
    if speaker_hits >= 2:
        return False, f"output has {speaker_hits} speaker tags"

    # 3. Input too short after cleaning (near-empty transcripts)
    if len(inp_clean.split()) < 20:
        return False, f"input too short after cleaning ({len(inp_clean.split())} words)"

    # 4. Output does not seem to be a coherent paragraph
    sentences = re.split(r'[.!?]+', out_clean)
    meaningful = [s.strip() for s in sentences if len(s.split()) >= 5]
    if len(meaningful) < 2:
        return False, "output has fewer than 2 meaningful sentences"

    return True, ""

# ─────────────────────────────────────────
# PROCESS ONE FILE
# ─────────────────────────────────────────
def process_file(in_path: str, out_path: str, split: str):
    total = kept = 0
    drop_reasons: dict[str, int] = {}

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            sample = json.loads(line)

            inp_clean = smart_truncate(clean_input(sample.get("input", "")))
            out_clean = clean_output(sample.get("output", ""))

            ok, reason = is_valid(inp_clean, out_clean)
            if not ok:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            fout.write(json.dumps({"input": inp_clean, "output": out_clean}) + "\n")
            kept += 1

    print(f"\n[{split}] {in_path}")
    print(f"  Total : {total}")
    print(f"  Kept  : {kept} ({kept/total*100:.1f}%)")
    print(f"  Dropped: {total-kept}")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {count}")
    print(f"  Saved → {out_path}")
    return kept

# ─────────────────────────────────────────
# QUICK VERIFY — print 2 cleaned samples
# ─────────────────────────────────────────
def verify(out_path: str, n: int = 2):
    print(f"\n===== SAMPLE VERIFICATION ({out_path}) =====")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            s = json.loads(line)
            print(f"\n--- Sample {i+1} ---")
            print(f"  INPUT  ({len(s['input'])} chars): {s['input'][:200]!r}")
            print(f"  OUTPUT ({len(s['output'].split())} words): {s['output'][:200]!r}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
print("=== DATA CLEANING PIPELINE ===")
print(f"Input  dir : {IN_DIR}")
print(f"Output dir : {OUT_DIR}")

n_train = process_file(TRAIN_IN, TRAIN_OUT, "TRAIN")
n_val   = process_file(VAL_IN,   VAL_OUT,   "VAL")

verify(TRAIN_OUT, n=2)

print(f"\n=== DONE ===")
print(f"Clean train : {n_train} samples → {TRAIN_OUT}")
print(f"Clean val   : {n_val}   samples → {VAL_OUT}")
print("\nNext step: run training.py")
