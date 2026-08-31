import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import SudokuSample, load_jsonl
from utils import *


def build_gemma4_chat_template():
    return (
        "{{ bos_token }}"
        "{%- macro strip_thinking(text) -%}"
        "{%- set ns = namespace(result='') -%}"
        "{%- for part in text.split('<channel|>') -%}"
        "{%- if '<|channel>' in part -%}"
        "{%- set ns.result = ns.result + part.split('<|channel>')[0] -%}"
        "{%- else -%}"
        "{%- set ns.result = ns.result + part -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{{ ns.result }}{%- endmacro -%}"
        "{% for message in messages %}"
        "{{ '<turn>' + message['role'] + '\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% elif message['content'] is iterable %}"
        "{% for c in message['content'] %}"
        "{% if c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<turn|>\n' }}"
        "{% endfor %}"
        "{{ '<turn>model\n' }}"
    )


def build_sample(tokenizer, prompt_text, solution_text):
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": solution_text},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text)
    if len(full_ids) > 512:
        return None

    start = full_text.find(solution_text)
    if start < 0:
        start = full_text.rfind("<turn>model\n")
        if start < 0:
            start = 0
        else:
            start += len("<turn>model\n")
    prompt_tokens = len(tokenizer.encode(full_text[:start]))
    labels = [-100] * len(full_ids)
    labels[prompt_tokens:] = full_ids[prompt_tokens:]
    return SudokuSample(full_ids, labels)


def make_dataset(tokenizer, records):
    out = []
    for rec in records:
        sample = build_sample(tokenizer, rec["prompt_text"], rec["solution_text"])
        if sample is not None:
            out.append(sample)
    return out


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(s.input_ids) for s in batch)
        input_ids = []
        attn = []
        labels = []
        for sample in batch:
            n = len(sample.input_ids)
            pad = max_len - n
            input_ids.append(sample.input_ids + [self.pad_token_id] * pad)
            attn.append([1] * n + [0] * pad)
            labels.append(sample.labels + [-100] * pad)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def load_model_and_lora():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", quantization_config=bnb, dtype=torch.float16)
    base = prepare_model_for_kbit_training(base)
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    return model


def evaluate(model, tokenizer, records):
    samples = make_dataset(tokenizer, records)
    if not samples:
        return math.inf

    samples = samples[:128]
    loader = DataLoader(samples, batch_size=4, shuffle=False, collate_fn=DataCollator(tokenizer.pad_token_id))

    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda")
            attn = torch.tensor(batch["attention_mask"], dtype=torch.long, device="cuda")
            labels = torch.tensor(batch["labels"], dtype=torch.long, device="cuda")
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            total += float(out.loss.detach().cpu())
            batches += 1
    model.train()
    if batches == 0:
        return math.inf
    return total / batches


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / "gemma-sudoku-lora"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SudokuDiffusion - Gemma QLoRA Fine-tuning")
    print("device = cuda")
    print("model = google/gemma-4-12B-it")
    print(f"checkpoint = {checkpoint_dir}")
    print("max_steps = 5000")
    print("batch x accum = 4x8")
    print("lr = 1e-4")
    print("=" * 60)

    tokenizer = load_tokenizer("google/gemma-4-12B-it")
    tokenizer.chat_template = build_gemma4_chat_template()
    model = load_model_and_lora()

    train_records = load_jsonl(ROOT / "artifacts" / "data" / "train.jsonl")
    val_records = load_jsonl(ROOT / "artifacts" / "data" / "val.jsonl", limit=50)
    train_samples = make_dataset(tokenizer, train_records)
    print(f"train samples: {len(train_samples):,}")
    print(f"val samples: {len(val_records):,}")

    loader = DataLoader(train_samples, batch_size=4, shuffle=True, collate_fn=DataCollator(tokenizer.pad_token_id), num_workers=0)

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    optim = AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=1e-4, betas=(0.9, 0.95))

    def lr_at(step):
        if step < 50:
            return 1e-4 * (step + 1) / 50
        progress = (step - 50) / (5000 - 50)
        return 1e-4 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    model.train()
    step = 0
    micro_step = 0
    best_eval_loss = -math.inf
    best_step = -1
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    optim.zero_grad(set_to_none=True)

    while step < 5000:
        for batch in loader:
            if step >= 5000:
                break

            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda")
            attn = torch.tensor(batch["attention_mask"], dtype=torch.long, device="cuda")
            labels = torch.tensor(batch["labels"], dtype=torch.long, device="cuda")

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / 8).backward()
            running_loss += float(out.loss.detach().cpu())
            running_count += 1
            micro_step += 1

            if micro_step % 8 == 0:
                for pg in optim.param_groups:
                    pg["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % 25 == 0:
                    avg = running_loss / max(1, running_count)
                    running_loss = 0.0
                    running_count = 0
                    print(f"step={step:5d}/5000 train_loss={avg:.4f} lr={optim.param_groups[0]['lr']:.2e} elapsed={time.time() - t0:.1f}s")

                if step % 32 == 0 or step == 5000:
                    eval_loss = evaluate(model, tokenizer, val_records)
                    print(f"eval step={step} eval_loss={eval_loss:.4f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_step = step
                        save_path = checkpoint_dir / "best"
                        save_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"saved best adapter to {save_path} eval_loss {eval_loss:.4f}")

    final_path = checkpoint_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    f = open(checkpoint_dir / "run_summary.json", "w")
    json.dump({"model_id": "google/gemma-4-12B-it", "max_steps": step, "best_step": best_step, "best_eval_loss": best_eval_loss}, f, indent=2)
    f.close()
