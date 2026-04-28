"""
evaluate_comparison.py — Kaggle-optimized evaluation script
============================================================
WHAT TO UPLOAD TO KAGGLE:
  1. This script (paste into a new cell in your Kaggle notebook)
  2. The fine-tuned adapter — if running in the SAME session as training,
     it's already at /kaggle/working/mistral-qlora/  (no upload needed)
     If running in a NEW session, download mistral-qlora/ folder and upload
     it as a Kaggle Dataset, then update ADAPTER_PATH below.
  3. Test data is already in your existing dataset (yakoob2345/processed)

PATHS — edit these two if needed:
"""

# ─────────────────────────────────────────
# INSTALL MISSING PACKAGES (Kaggle doesn't pre-install these)
# ─────────────────────────────────────────
import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("evaluate")
pip_install("bert_score")
pip_install("rouge_score")   # evaluate's rouge backend
pip_install("sacrebleu")     # evaluate's bleu backend

import json
import os
import re

import evaluate
import pandas as pd
import torch
from bert_score import score as bert_score_func
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.manual_seed(42)

# ─────────────────────────────────────────
# KAGGLE PATHS  ← only edit these
# ─────────────────────────────────────────
TEST_FILE    = "/kaggle/input/datasets/yakoob2345/processed/test.jsonl"
ADAPTER_PATH = "/kaggle/working/mistral-qlora"       # output dir from training.py
OUTPUT_CSV   = "/kaggle/working/evaluation_results.csv"

# Optional — set to None if you don't have pre-computed baseline files
ZERO_SHOT_FILE   = None   # e.g. "/kaggle/input/.../zero_shot_preds.json"
PROMPT_PRED_FILE = None   # e.g. "/kaggle/input/.../prompt_preds.json"

MODEL_NAME      = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_INPUT_CHARS = 1000   # must match training INPUT_CHAR_LIMIT
MAX_NEW_TOKENS  = 180    # raised: more room to capture key details → better ROUGE-L
N_TEST_SAMPLES  = 50     # how many test samples to evaluate
BATCH_SIZE      = 4      # batched inference: 4x faster than one-by-one on T4
RUN_BERTSCORE   = True   # set False to skip BERTScore and save ~3 min

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────
# ACTION ITEM EXTRACTION
# ─────────────────────────────────────────
def extract_action_items(text):
    bullets = re.findall(r'(?:^|\n)[ \t]*[*\-•\d.]+[ \t]+([^\n]+)', text)
    if bullets:
        return [b.strip() for b in bullets]
    keywords = ['should', 'will', 'needs to', 'to do', 'action item', 'decided to', 'agreed to']
    return [s.strip() for s in re.split(r'[.!?]\s+', text) if any(k in s.lower() for k in keywords)]

def compute_action_item_f1(preds, targets):
    scores = []
    for p, t in zip(preds, targets):
        p_items = " ".join(extract_action_items(p)).lower().split()
        t_items = " ".join(extract_action_items(t)).lower().split()
        if not p_items and not t_items:
            scores.append(1.0); continue
        if not p_items or not t_items:
            scores.append(0.0); continue
        inter = set(p_items) & set(t_items)
        pr = len(inter) / len(set(p_items))
        rc = len(inter) / len(set(t_items))
        scores.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return sum(scores) / len(scores)

# ─────────────────────────────────────────
# INPUT CLEANING  (identical to training.py)
# ─────────────────────────────────────────
def clean_input(text):
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
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def smart_truncate(text, max_chars=MAX_INPUT_CHARS):
    """Keep start + end — meeting conclusions are at the end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]

# INSTRUCTION — must match training.py + extra anti-hallucination clauses at inference
# ─────────────────────────────────────────
INSTRUCTION = (
    "You are given a cleaned meeting transcript.\n\n"
    "Your task is to produce a high-quality summary that captures:\n"
    "1. the main topics discussed\n"
    "2. the key decisions or agreements reached\n"
    "3. the final outcomes or conclusions\n\n"
    "STRICT RULES:\n"
    "- Use ONLY information from the transcript below\n"
    "- Do NOT add external knowledge or guess missing details\n"
    "- Do NOT include speaker names or dialogue fragments\n"
    "- Do NOT repeat conversational details or filler content\n"
    "- Keep the summary concise, factual, and coherent\n\n"
    "Write the summary as a single well-structured paragraph."
)

def build_prompt(text):
    """
    Prompt must match training.py format exactly.
    Extra anti-hallucination clauses are safe to add at inference time—
    the model was trained on the ### Instruction / ### Input / ### Response structure
    so any instruction in that block will be followed.
    """
    cleaned = smart_truncate(clean_input(text))
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{cleaned}\n\n"
        "### Response:\n"
    )

# ─────────────────────────────────────────
# OUTPUT CLEANING
# ─────────────────────────────────────────
def clean_output(decoded, prompt):
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.replace(prompt, "").replace("### Instruction:", "") \
                  .replace("### Input:", "").replace("### Response:", "").strip()

# ─────────────────────────────────────────
# GENERATION — batched for speed
# Single-sample inference on T4 wastes ~70% of GPU bandwidth;
# batching 4 at a time gives ~3-4x speedup with no quality loss.
# ─────────────────────────────────────────
tokenizer_for_batch = None   # set during run_evaluation after tokenizer is loaded

def generate_batch(model, tokenizer, prompts):
    """Run inference on a list of prompts in one GPU call."""
    tokenizer.padding_side = "left"   # left-pad for decoder-only batch generation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=768,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.3,
            length_penalty=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        # Strip the input tokens — only decode new tokens
        new_tokens = out[inputs["input_ids"].shape[1]:]
        decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred       = decoded.strip()
        results.append(pred if pred else "No summary generated.")
    return results

# ─────────────────────────────────────────
# HALLUCINATION GUARD
# Checks whether the model's prediction is grounded in the input.
# Words not in the input AND not common English connectives count as novel.
# If too many novel words appear the model is hallucinating — regenerate once.
# ─────────────────────────────────────────
_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall','can',
    'it','its','this','that','these','those','i','we','you','he','she','they',
    'who','which','what','when','where','why','how','and','or','but','if','as',
    'at','by','for','in','of','on','to','with','about','after','before','into',
    'not','no','nor','so','yet','also','then','just','more','most','some','all',
    # domain words the model may legitimately generate even if not verbatim in input
    'meeting','discussed','team','group','decided','agreed','concluded','noted',
    'proposed','suggested','participants','members','during','session','focused',
    'including','regarding','related','based','key','main','overall','summary',
}

def is_grounded(pred: str, input_text: str, threshold: int = 25) -> bool:
    """Return True if the prediction is mostly grounded in the input."""
    input_words = set(input_text.lower().split())
    pred_words  = set(pred.lower().split())
    novel = pred_words - input_words - _STOPWORDS
    return len(novel) < threshold

def regenerate_single(model, tokenizer, prompt: str) -> str:
    """Fallback single-sample regeneration with slight temperature for variety."""
    tokenizer.padding_side = "right"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=768).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,            # slight randomness for different output
            temperature=0.7,
            repetition_penalty=1.3,
            length_penalty=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return pred if pred else "No summary generated."

def run_inference(model, tokenizer, test_data):
    """Batched inference + hallucination guard with one re-generation pass."""
    prompts = [build_prompt(item["input"]) for item in test_data]
    preds   = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
        batch  = prompts[i : i + BATCH_SIZE]
        preds += generate_batch(model, tokenizer, batch)

    # Hallucination guard: check each prediction; regenerate once if ungrounded
    regen_count = 0
    for j, (pred, item) in enumerate(zip(preds, test_data)):
        if not is_grounded(pred, item["input"]):
            preds[j]    = regenerate_single(model, tokenizer, prompts[j])
            regen_count += 1
    if regen_count:
        print(f"  Re-generated {regen_count}/{len(preds)} samples (hallucination guard)")
    return preds

# ─────────────────────────────────────────
# METRICS — load once, reuse for all models
# ─────────────────────────────────────────
_rouge = None
_bleu  = None

def compute_metrics(preds, refs, label):
    global _rouge, _bleu
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    if _bleu is None:
        _bleu = evaluate.load("bleu")

    print(f"  ROUGE/BLEU: {label}")
    r  = _rouge.compute(predictions=preds, references=refs)
    b  = _bleu.compute( predictions=preds, references=[[x] for x in refs])

    result = {
        "ROUGE-1":        round(r["rouge1"] * 100, 2),
        "ROUGE-2":        round(r["rouge2"] * 100, 2),
        "ROUGE-L":        round(r["rougeL"] * 100, 2),
        "BLEU-4":         round(b["bleu"]   * 100, 2),
        "Action-Item-F1": round(compute_action_item_f1(preds, refs) * 100, 2),
    }

    if RUN_BERTSCORE:
        print(f"  BERTScore : {label}")
        _, _, f1 = bert_score_func(preds, refs, lang="en", verbose=False)
        result["BERTScore-F1"] = round(f1.mean().item() * 100, 2)

    return result

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_evaluation():
    # ── Load test data ──
    print(f"Loading test data from: {TEST_FILE}")
    with open(TEST_FILE) as f:
        test_data = [json.loads(x) for x in f][:N_TEST_SAMPLES]
    refs = [x["output"] for x in test_data]
    print(f"Evaluating on {len(test_data)} samples")

    # ── Load model ──
    print(f"\nLoading base model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading adapter from: {ADAPTER_PATH}")
    # Auto-detect v2 output folder if ADAPTER_PATH doesn't exist
    _adapter = ADAPTER_PATH
    if not os.path.exists(_adapter):
        for fallback in ["/kaggle/working/mistral-qlora-v2",
                          "/kaggle/working/mistral-qlora",
                          "./mistral-qlora-v2",
                          "./mistral-qlora"]:
            if os.path.exists(fallback):
                _adapter = fallback
                print(f"  (Using fallback: {_adapter})")
                break
    assert os.path.exists(_adapter), (
        f"Adapter not found at any known path. Last tried: {_adapter}\n"
        "Upload the mistral-qlora-v2/ folder as a Kaggle Dataset and update ADAPTER_PATH."
    )
    model = PeftModel.from_pretrained(base_model, _adapter)
    model.eval()
    print("Model ready.\n")

    # ── Run fine-tuned inference (batched) ──
    print(f"Running batched inference (batch_size={BATCH_SIZE})...")
    import time
    t0 = time.time()
    ft_preds = run_inference(model, tokenizer, test_data)
    print(f"Inference done in {time.time()-t0:.0f}s")

    # ── Print sample output ──
    print("\n--- SAMPLE OUTPUT (first example) ---")
    print(f"INPUT   : {test_data[0]['input'][:200]}")
    print(f"GT      : {refs[0]}")
    print(f"FT PRED : {ft_preds[0]}")

    # ── Collect models to compare ──
    models_to_eval = {"Fine-tuned (Mistral-7B)": ft_preds}

    if ZERO_SHOT_FILE and os.path.exists(ZERO_SHOT_FILE):
        with open(ZERO_SHOT_FILE) as f:
            models_to_eval["Zero-shot"] = json.load(f)[:N_TEST_SAMPLES]
    else:
        print("\n[INFO] No zero-shot predictions file — skipping baseline comparison.")

    if PROMPT_PRED_FILE and os.path.exists(PROMPT_PRED_FILE):
        with open(PROMPT_PRED_FILE) as f:
            models_to_eval["Prompt-engineered"] = json.load(f)[:N_TEST_SAMPLES]

    # ── Compute metrics ──
    print("\nComputing metrics...")
    results = {}
    for name, preds in models_to_eval.items():
        results[name] = compute_metrics(preds, refs, name)

    # ── Print results ──
    df = pd.DataFrame(results).T
    print("\n========== EVALUATION RESULTS ==========")
    print(df.to_string())
    print("=========================================")

    ft_rl = results["Fine-tuned (Mistral-7B)"]["ROUGE-L"]
    print(f"\nFine-tuned ROUGE-L : {ft_rl:.2f}")
    if ft_rl >= 20:
        print("✅ GOOD — model learned to summarize")
    elif ft_rl >= 12:
        print("⚠️  MODERATE — model is summarizing but weakly")
    else:
        print("❌ POOR — review training pipeline")

    # ── Save ──
    df.to_csv(OUTPUT_CSV)
    print(f"\nResults saved to: {OUTPUT_CSV}")

    # ── Save predictions ──
    pred_out = "/kaggle/working/ft_predictions.json"
    with open(pred_out, "w") as f:
        json.dump(ft_preds, f, indent=2)
    print(f"Predictions saved to: {pred_out}")

    # ── Save qualitative examples (for report / examiner analysis) ──
    N_QUAL = min(5, len(test_data))
    qual_examples = []
    for i in range(N_QUAL):
        inp    = test_data[i]["input"]
        gt     = refs[i]
        pred   = ft_preds[i]
        grnd   = is_grounded(pred, inp)
        qual_examples.append({
            "example": i + 1,
            "input_snippet": inp[:300] + ("..." if len(inp) > 300 else ""),
            "ground_truth":  gt,
            "model_output":  pred,
            "grounded":      grnd,
            "analysis": (
                "Grounded: model output aligns with transcript content."
                if grnd else
                "Ungrounded: model may have introduced external information."
            ),
        })
    qual_out = "/kaggle/working/qualitative_examples.json"
    with open(qual_out, "w") as f:
        json.dump(qual_examples, f, indent=2, ensure_ascii=False)
    print(f"Qualitative examples saved to: {qual_out}")

    # Pretty-print qualitative examples
    print("\n====== QUALITATIVE ANALYSIS (first 3 examples) ======")
    for ex in qual_examples[:3]:
        print(f"\n--- Example {ex['example']} ({'GROUNDED' if ex['grounded'] else 'UNGROUNDED'}) ---")
        print(f"  INPUT   : {ex['input_snippet'][:150]}")
        print(f"  GT      : {ex['ground_truth'][:200]}")
        print(f"  PRED    : {ex['model_output'][:200]}")
        print(f"  ANALYSIS: {ex['analysis']}")
    print("=====================================================")


if __name__ == "__main__":
    run_evaluation()