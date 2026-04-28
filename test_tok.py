import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

INPUT_CHAR_LIMIT = 1200
MAX_LENGTH = 1024

def clean_input(text):
    text = text.replace("Summarize the following meeting:", "")
    return text.strip()

def tokenize_function(example):
    cleaned_input = clean_input(example["input"])
    prompt = f"### Instruction:\nSummarize the meeting.\n\n### Input:\n{cleaned_input[:INPUT_CHAR_LIMIT]}\n\n### Response:\n"
    
    full_text = prompt + example["output"] + tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    labels = tokenized["input_ids"].copy()
    
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )["input_ids"]
    
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    tokenized["labels"] = labels
    return tokenized

example = {
    "input": "Hello world " * 100,
    "output": "The meeting was about hello world."
}
out = tokenize_function(example)
print("Total length:", len(out["input_ids"]))
pad_id = tokenizer.pad_token_id
pad_count = out["input_ids"].count(pad_id)
print("Pad count in input_ids:", pad_count)
pad_labels = [l for t, l in zip(out["input_ids"], out["labels"]) if t == pad_id]
print("Labels for pad tokens:", set(pad_labels))
