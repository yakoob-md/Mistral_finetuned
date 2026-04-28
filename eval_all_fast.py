import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_CHARS = 2000
MAX_NEW_TOKENS = 256

def build_prompt(text):
    text = text[:MAX_INPUT_CHARS]
    return f"### Instruction:\nSummarize the meeting and extract key discussion points.\n\n### Input:\n{text}\n\n### Response:\n"

def clean_output(decoded, prompt):
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.strip()

with open("data/processed/test.jsonl") as f:
    test_data = [json.loads(x) for x in f][:1]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

checkpoints = [
    "outputs/checkpoints/checkpoint-600",
    "outputs/checkpoints/checkpoint-625",
    "outputs/checkpoints/phi2-150mb",
    "outputs/checkpoints/phi2-80mb",
    "outputs/checkpoints/phi2-qlora"
]

for ckpt in checkpoints:
    print(f"\nEvaluating {ckpt}...")
    try:
        model = PeftModel.from_pretrained(base_model, ckpt)
        model.eval()
        
        prompt = build_prompt(test_data[0]["input"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = clean_output(decoded, prompt)
        print(f"SAMPLE OUTPUT: {res}")
        del model
    except Exception as e:
        print(f"Error evaluating {ckpt}: {e}")
