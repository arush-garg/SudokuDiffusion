import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fine_tuning.gemma import build_gemma4_chat_template


def load_model_and_adapter(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = build_gemma4_chat_template()

    base = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model = model.to("mps")
    model.eval()
    return model, tokenizer


def run_inference(checkpoint_path, prompt_text):
    model, tokenizer = load_model_and_adapter(checkpoint_path)
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to("mps")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
