# =========================================
# FINAL DATASET PIPELINE (CLEAN + CORRECT)
# =========================================

import json
import random
from datasets import load_dataset
from collections import defaultdict

SEED = 42
random.seed(SEED)

# -------------------------------
# CHUNKING (FIXED)
# -------------------------------

def chunk_text(text, max_words=600):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])


# -------------------------------
# SIMPLE EXTRACTIVE SUMMARY (AMI FIX)
# -------------------------------

def simple_extractive_summary(text, n_sentences=3):
    sentences = text.split(".")
    return ". ".join(sentences[:n_sentences]).strip()


# -------------------------------
# LOAD DATASETS
# -------------------------------

print("Loading AMI...")
ami = load_dataset("edinburghcstr/ami", "ihm", split="train")

print("Loading QMSum...")
qmsum = load_dataset(
    "json",
    data_files="/kaggle/input/datasets/arjunnm/qmsum-data/jsonl/train.jsonl",
    split="train"
)


# -------------------------------
# PROCESS QMSUM (FIXED)
# -------------------------------

def process_qmsum(dataset):
    samples = []

    for example in dataset:
        meeting_id = example.get("meeting_id", str(random.random()))

        transcript = "\n".join([
            f"{t.get('speaker','Unknown')}: {t.get('content','')}"
            for t in example.get("meeting_transcripts", [])
        ])

        # reduce size (IMPORTANT)
        short_transcript = " ".join(transcript.split()[:600])

        # GENERAL QUERIES
        for q in example.get("general_query_list", []):
            samples.append({
                "meeting_id": meeting_id,
                "input": f"Summarize the following meeting:\n{short_transcript}",
                "output": q.get("answer", "")
            })

        # SPECIFIC QUERIES
        for q in example.get("specific_query_list", []):
            samples.append({
                "meeting_id": meeting_id,
                "input": f"Answer the query: {q.get('query','')}\nMeeting:\n{short_transcript}",
                "output": q.get("answer", "")
            })

    return samples


# -------------------------------
# PROCESS AMI (CORRECT FOR YOUR FORMAT)
# -------------------------------

def process_ami(dataset):
    samples = []
    meetings = defaultdict(list)

    # Group by meeting
    for row in dataset:
        meeting_id = row.get("meeting_id", "unknown")

        text = row.get("text", "").strip()
        speaker = row.get("speaker_id", "Unknown")

        if text:
            meetings[meeting_id].append(f"{speaker}: {text}")

    # Build samples
    for meeting_id, utterances in meetings.items():

        full_text = "\n".join(utterances)

        if not full_text.strip():
            continue

        for chunk in chunk_text(full_text, max_words=600):

            samples.append({
                "meeting_id": meeting_id,
                "input": f"Summarize the following meeting segment:\n{chunk}",
                "output": simple_extractive_summary(chunk)  # 🔥 FIXED
            })

    return samples


# -------------------------------
# RUN PROCESSING
# -------------------------------

print("Processing QMSum...")
qmsum_samples = process_qmsum(qmsum)
print(f"QMSum samples: {len(qmsum_samples)}")

print("Processing AMI...")
ami_samples = process_ami(ami)
print(f"AMI samples: {len(ami_samples)}")

data = qmsum_samples + ami_samples
print(f"Total samples: {len(data)}")


# -------------------------------
# SPLIT BY MEETING_ID (NO LEAKAGE)
# -------------------------------

from collections import defaultdict
import random

def split_by_meeting_id(data, seed=42):
    random.seed(seed)

    # Group by meeting_id
    meeting_groups = defaultdict(list)
    for item in data:
        meeting_groups[item["meeting_id"]].append(item)

    meeting_ids = list(meeting_groups.keys())
    random.shuffle(meeting_ids)

    n = len(meeting_ids)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_ids = set(meeting_ids[:train_end])
    val_ids = set(meeting_ids[train_end:val_end])
    test_ids = set(meeting_ids[val_end:])

    train, val, test = [], [], []

    for mid, items in meeting_groups.items():
        if mid in train_ids:
            train.extend(items)
        elif mid in val_ids:
            val.extend(items)
        else:
            test.extend(items)

    return train, val, test


train_data, val_data, test_data = split_by_meeting_id(data)

print("\nSplit sizes:")
print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")


# -------------------------------
# CHECK MAX INPUT LENGTH
# -------------------------------

max_len = max(len(x["input"].split()) for x in data)
print(f"\nMax tokens: {max_len}")


# -------------------------------
# SAVE JSONL
# -------------------------------

def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

save_jsonl("train.jsonl", train_data)
save_jsonl("val.jsonl", val_data)
save_jsonl("test.jsonl", test_data)

print("\nSaved: train.jsonl, val.jsonl, test.jsonl")