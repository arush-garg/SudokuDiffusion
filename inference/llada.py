import argparse
import json
import math
import sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_model_for_inference(checkpoint_path=None):
    checkpoint_dir = Path(checkpoint_path) if checkpoint_path else ROOT / "artifacts" / "checkpoints" / "illada-sudoku-lora"
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/iLLaDA-8B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained("GSAI-ML/iLLaDA-8B-Base", dtype=torch.float16, trust_remote_code=True)
    base.config.use_cache = False

    if checkpoint_dir.exists() and (checkpoint_dir / "adapter_model.safetensors").exists():
        print(f"Attaching LoRA adapter from {checkpoint_dir}")
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    else:
        print(f"no LoRA at {checkpoint_dir}; using base model")
        model = base

    model.to("mps")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt_text, num_steps=12, unmask_k=1):
    target_canvas_len = 199
    if unmask_k > 1:
        num_steps = math.ceil(target_canvas_len / unmask_k)

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = 5
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("empty prompt")

    canvas = torch.full(
        (1, len(prompt_ids) + target_canvas_len),
        fill_value=mask_id,
        dtype=torch.long,
        device="mps",
    )
    canvas[0, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device="mps")
    locked = torch.zeros(canvas.shape[1], dtype=torch.bool, device="mps")
    locked[:len(prompt_ids)] = True

    with torch.no_grad():
        for step in range(num_steps):
            outputs = model(input_ids=canvas, attention_mask=torch.ones_like(canvas), use_cache=False)
            probs = torch.softmax(outputs.logits, dim=-1)
            conf, preds = probs.max(dim=-1)
            conf = conf.squeeze(0)
            preds = preds.squeeze(0)
            conf[locked.clone()] = -1.0

            remaining = int((~locked).sum().item())
            if remaining <= 0:
                break
            if step == num_steps - 1:
                k = remaining
            elif unmask_k == 1:
                k = max(1, target_canvas_len // num_steps)
            else:
                k = unmask_k

            topk = torch.topk(conf, k=min(k, remaining)).indices
            canvas[0, topk] = preds[topk]
            locked[topk] = True

    target_ids = canvas[0, len(prompt_ids):].tolist()
    return tokenizer.decode(target_ids, skip_special_tokens=True)


def run_inference(checkpoint_path=None, prompt_text="", num_steps=12):
    model, tokenizer = load_model_for_inference(checkpoint_path)
    return generate(model, tokenizer, prompt_text, num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "checkpoints" / "illada-sudoku-lora"))
    parser.add_argument("--num_steps", type=int, default=12)
    args = parser.parse_args()
    text = run_inference(args.checkpoint, args.prompt, args.num_steps)
    print(json.dumps({"prompt": args.prompt, "num_steps": args.num_steps, "text": text}, indent=2))
