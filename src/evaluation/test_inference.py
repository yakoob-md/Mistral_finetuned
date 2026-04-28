
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

def clean_output(decoded, prompt):
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    
    # fallback: remove prompt parts manually if model over-copied
    cleaned = decoded.replace(prompt, "")
    cleaned = cleaned.replace("### Instruction:", "")
    cleaned = cleaned.replace("### Input:", "")
    cleaned = cleaned.replace("### Response:", "")
    
    return cleaned.strip()

def test_inference():
    model_name = "microsoft/phi-2"
    adapter_path = "outputs/checkpoints/phi2-qlora"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print(f"Loading adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Test Input
    test_input = "Project Manager: Okay, so we're here to discuss the new project timeline. Marketing: We need at least 3 weeks for the campaign setup. Designer: I can finish the UI in 1 week if the assets are ready. Project Manager: Great, so let's aim for a launch in 4 weeks."
    
    # Match training prompt
    prompt = f"""### Instruction:
Summarize the meeting and extract key discussion points.

### Input:
{test_input}

### Response:
"""
    
    print("\n--- RUNNING INFERENCE ---")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.3,   # 🔥 increased to prevent prompt copying
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_output(decoded, prompt)
        
    print("\nPROMPT:\n", prompt)
    print("\nGENERATED SUMMARY (CLEANED):\n", response)
    print("\n-------------------------")

if __name__ == "__main__":
    test_inference()
