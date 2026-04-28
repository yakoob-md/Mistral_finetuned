# =========================================
# MISTRAL-7B QLORA TRAINING v2 — PRODUCTION
# =========================================
# Run clean_data.py FIRST to produce:
#   /kaggle/working/clean_train.jsonl
#   /kaggle/working/clean_val.jsonl
#
# All v2 improvements:
# [x] Loads pre-cleaned data (speaker tags / filler / noise already removed)
# [x] Better instruction — focuses on decisions/outcomes
# [x] Proper HF Trainer + DataCollatorForSeq2Seq (NOT SFTTrainer)
# [x] Correct label masking — loss only on response tokens
# [x] EarlyStoppingCallback (patience=2) — stops before overfitting
# [x] PrintLossCallback — visible logs in Kaggle notebooks
# [x] eval_strategy="steps" every 50 steps — catches overfitting early
# [x] enable_input_require_grads() — required for LoRA + gradient checkpointing
# [x] Zip output for easy Kaggle download
# [x] Masking verification prints before training
# =========================================

import os, re, shutil
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Cleaned data — output from clean_data.py
KAGGLE_CLEAN = "/kaggle/working"
LOCAL_CLEAN  = "./data/processed"
CLEAN_DIR    = KAGGLE_CLEAN if os.path.exists("/kaggle/working/clean_train.jsonl") else LOCAL_CLEAN

TRAIN_FILE = f"{CLEAN_DIR}/clean_train.jsonl"
VAL_FILE   = f"{CLEAN_DIR}/clean_val.jsonl"
OUTPUT_DIR = "/kaggle/working/mistral-qlora-v2"

MAX_SAMPLES = 2000
MAX_LENGTH  = 768    # token budget — safe on T4 with Mistral-7B 4-bit

print(f"Train data : {TRAIN_FILE}")
print(f"Val data   : {VAL_FILE}")
print(f"Model      : {MODEL_NAME}")

# ─────────────────────────────────────────
# LOGGING CALLBACK — forces loss to print in Kaggle notebooks
# HuggingFace Trainer logs internally but doesn't flush to notebook stdout
# ─────────────────────────────────────────
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step    = state.global_step
        loss    = logs.get("loss", "")
        ev_loss = logs.get("eval_loss", "")
        lr      = logs.get("learning_rate", "")
        parts   = [f"Step {step:>4}"]
        if loss:    parts.append(f"train_loss={float(loss):.4f}")
        if ev_loss: parts.append(f"eval_loss={float(ev_loss):.4f}")
        if lr:      parts.append(f"lr={float(lr):.2e}")
        # Overfitting hint — eval >> train means memorisation
        if loss and ev_loss and float(ev_loss) > float(loss) * 1.25:
            parts.append("⚠ OVERFIT RISK")
        print(" | ".join(parts), flush=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("\nLoading cleaned data...")
assert os.path.exists(TRAIN_FILE), (
    f"Clean train file not found: {TRAIN_FILE}\n"
    "Run clean_data.py first!"
)
raw = load_dataset("json", data_files={"train": TRAIN_FILE, "val": VAL_FILE})
raw["train"] = raw["train"].shuffle(seed=42).select(range(min(MAX_SAMPLES, len(raw["train"]))))
print(f"Train: {len(raw['train'])} | Val: {len(raw['val'])}")

# ─────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────
# PROMPT TEMPLATE
# Stronger instruction — focuses model on decisions/outcomes
# (NOT just "Summarize the meeting" which is too vague)
# ─────────────────────────────────────────
INSTRUCTION = (
    "Summarize the key discussions, decisions, and outcomes of the meeting. "
    "Focus only on the most important points. Be concise and avoid filler content."
)

def build_prompt(input_text: str) -> str:
    # Data is already cleaned by clean_data.py — no need to re-clean here
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    )

# ─────────────────────────────────────────
# TOKENIZATION + LABEL MASKING
# Loss is ONLY computed on response tokens (labels = -100 for prompt)
# ─────────────────────────────────────────
def tokenize_function(example):
    prompt   = build_prompt(example["input"])
    response = example["output"] + tokenizer.eos_token
    full     = prompt + response

    full_enc = tokenizer(full,   truncation=True, max_length=MAX_LENGTH, padding=False)
    prmt_enc = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding=False)
    prompt_len = len(prmt_enc["input_ids"])

    labels = [-100] * len(full_enc["input_ids"])
    labels[prompt_len:] = full_enc["input_ids"][prompt_len:]

    if all(l == -100 for l in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    full_enc["labels"] = labels
    return full_enc

print("Tokenizing...")
train_tok = raw["train"].map(tokenize_function, remove_columns=raw["train"].column_names, batched=False)
val_tok   = raw["val"].map(  tokenize_function, remove_columns=raw["val"].column_names,   batched=False)

train_tok = train_tok.filter(lambda x: len(x["input_ids"]) > 0 and any(l != -100 for l in x["labels"]))
val_tok   = val_tok.filter(  lambda x: len(x["input_ids"]) > 0 and any(l != -100 for l in x["labels"]))
print(f"Train after filter: {len(train_tok)} | Val: {len(val_tok)}")

# ─────────────────────────────────────────
# MASKING VERIFICATION — confirm labels are correct before wasting GPU time
# ─────────────────────────────────────────
print("\n===== MASKING VERIFICATION =====")
for i in range(min(3, len(train_tok))):
    s        = train_tok[i]
    total    = len(s["input_ids"])
    masked   = sum(1 for l in s["labels"] if l == -100)
    unmasked = total - masked
    resp_ids = [t for t, l in zip(s["input_ids"], s["labels"]) if l != -100]
    resp_txt = tokenizer.decode(resp_ids, skip_special_tokens=True)
    print(f"  Sample {i+1}: total={total} | prompt(masked)={masked} | response(unmasked)={unmasked}")
    print(f"    Response: {resp_txt[:100]!r}")
    if unmasked == 0:
        print("    *** WARNING: all tokens masked — check truncation! ***")
print("=================================\n")

# ─────────────────────────────────────────
# QLoRA CONFIG
# ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Both required: LoRA + gradient checkpointing need input grads enabled
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ─────────────────────────────────────────
# LoRA — Mistral-7B target modules
# ─────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────
# DATA COLLATOR — pads labels with -100, not 0
# ─────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ─────────────────────────────────────────
# TRAINING ARGS
#
# HOW TO DETECT OVERFITTING in the logs:
#   train_loss drops steadily BUT eval_loss plateaus or rises → overfit
#   EarlyStoppingCallback will auto-stop after 2 eval steps with no improvement
#   Manual stop: interrupt if eval_loss hasn't dropped in last 3-4 prints
# ─────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,     # effective batch = 8

    learning_rate=1e-4,
    num_train_epochs=4,
    warmup_steps=50,
    lr_scheduler_type="cosine",

    # Frequent eval/save — catches overfitting before step 200
    logging_steps=10,
    logging_first_step=True,
    save_steps=50,
    save_total_limit=3,
    eval_strategy="steps",             # "evaluation_strategy" is deprecated
    eval_steps=50,

    # Best model tracking (required by EarlyStoppingCallback)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=True,                         # Tensor Cores on T4 — 2-3x faster
    bf16=False,

    optim="paged_adamw_8bit",
    max_grad_norm=1.0,                 # gradient clipping
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to="none",
)

# ─────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    callbacks=[
        PrintLossCallback(),
        EarlyStoppingCallback(early_stopping_patience=2),
    ],
)

# ─────────────────────────────────────────
# PRE-TRAINING INFERENCE CHECK
# ─────────────────────────────────────────
print("\n===== PRE-TRAINING INFERENCE CHECK =====")
model.eval()
sample  = train_tok[0]
ids     = torch.tensor([sample["input_ids"]]).to(model.device)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, do_sample=False)
pre_txt = tokenizer.decode(out[0][len(sample["input_ids"]):], skip_special_tokens=True)
print(f"Zero-shot preview: {pre_txt[:200]!r}")
print("=========================================\n")
model.train()

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
print("===== TRAINING START =====")
torch.cuda.empty_cache()
trainer.train()

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
print("\nSaving best adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ─────────────────────────────────────────
# ZIP FOR DOWNLOAD
# ─────────────────────────────────────────
zip_path = "/kaggle/working/mistral-qlora-v2"
print(f"\nZipping to {zip_path}.zip ...")
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)

print("\n===== DONE =====")
print(f"Adapter   : {OUTPUT_DIR}")
print(f"Zip       : {zip_path}.zip")
print(f"Steps run : {trainer.state.global_step}")
if trainer.state.best_metric is not None:
    print(f"Best eval loss: {trainer.state.best_metric:.4f}")