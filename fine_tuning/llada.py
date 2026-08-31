import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import SudokuDataset
from utils import *


class Batch:
    def __init__(self, input_ids, labels, attention_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collator(tokenizer):
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = 5
    eos_id = tokenizer.eos_token_id

    def collate(records):
        if not records:
            return None

        ids_batch = []
        labels_batch = []
        attn_batch = []
        for rec in records:
            prompt_ids = tokenizer.encode(rec["prompt_text"], add_special_tokens=False)
            solution_ids = tokenizer.encode(rec["solution_text"], add_special_tokens=False)
            if len(solution_ids) >= 199:
                solution_ids = solution_ids[:199]
            else:
                solution_ids = solution_ids + [eos_id] * (199 - len(solution_ids))

            full_ids = prompt_ids + solution_ids
            prompt_len = len(prompt_ids)
            n_mask = max(0, min(int(math.floor(random.random() * len(solution_ids))), len(solution_ids)))
            positions = list(range(prompt_len, prompt_len + len(solution_ids)))
            random.shuffle(positions)
            mask_positions = set(positions[:n_mask])

            labels = [-100] * len(full_ids)
            input_ids = list(full_ids)
            for pos in mask_positions:
                labels[pos] = full_ids[pos]
                input_ids[pos] = mask_id

            ids_batch.append(input_ids)
            labels_batch.append(labels)
            attn_batch.append([1] * len(input_ids))

        max_len = max(len(x) for x in ids_batch)
        for i in range(len(ids_batch)):
            pad = max_len - len(ids_batch[i])
            ids_batch[i] = ids_batch[i] + [eos_id] * pad
            labels_batch[i] = labels_batch[i] + [-100] * pad
            attn_batch[i] = attn_batch[i] + [0] * pad
        return Batch(ids_batch, labels_batch, attn_batch)

    return collate


def load_model_and_tokenizer():
    tokenizer = load_tokenizer("GSAI-ML/iLLaDA-8B-Base", trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/iLLaDA-8B-Base",
        quantization_config=bnb,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    model.to("cuda")
    return model, tokenizer


def compute_loss(model, batch):
    input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device="cuda")
    labels = torch.tensor(batch.labels, dtype=torch.long, device="cuda")
    attn = torch.tensor(batch.attention_mask, dtype=torch.long, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    return torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)


def load_adapter_weights(model, path):
    raw = load_file(str(path / "adapter_model.safetensors"))
    adapter = getattr(model, "active_adapter", "default")
    remapped = {}
    for key, tensor in raw.items():
        if ".lora_A.weight" in key:
            new_key = key.replace(".lora_A.weight", f".lora_A.{adapter}.weight")
        elif ".lora_B.weight" in key:
            new_key = key.replace(".lora_B.weight", f".lora_B.{adapter}.weight")
        else:
            new_key = key
        remapped[new_key] = tensor.to(model.device)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"loaded {len(remapped)} LoRA tensors from {path}, missing={len(missing)}, unexpected={len(unexpected)}")


def save_checkpoint(model, tokenizer, step):
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / "illada-sudoku-lora"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    f = open(checkpoint_dir / "training_meta.json", "w")
    json.dump({"TARGET_CANVAS_LEN": 199, "step": step, "model_name": "GSAI-ML/iLLaDA-8B-Base"}, f, indent=2)
    f.close()


if __name__ == "__main__":
    set_seed(42)
    print("TARGET_CANVAS_LEN=199")

    model, tokenizer = load_model_and_tokenizer()
    train_ds = SudokuDataset(ROOT / "artifacts" / "data" / "train.jsonl")
    val_ds = SudokuDataset(ROOT / "artifacts" / "data" / "val.jsonl")
    val_records = val_ds.records[:50]
    print(f"Loaded train={len(train_ds)} val={len(val_ds)} eval={len(val_records)}")

    collate = make_collator(tokenizer)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < 50:
            return step / 50
        progress = (step - 50) / (5000 - 50)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / "illada-sudoku-lora"
    metrics_path = checkpoint_dir / "metrics.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    step = 0

    if step == 0 and (checkpoint_dir / "adapter_model.safetensors").exists():
        print(f"warm starting adapter from {checkpoint_dir}")
        load_adapter_weights(model, checkpoint_dir)

    optim.zero_grad(set_to_none=True)
    loss_for_log = 0.0

    while step < 5000:
        idxs = [rng.randrange(len(train_ds)) for _ in range(1)]
        batch = collate([train_ds[i] for i in idxs])
        if batch is None:
            continue

        model.train()
        loss = compute_loss(model, batch)
        loss_for_log = float(loss.item())
        (loss / 8).backward()

        if (step + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        step += 1

        if step % 10 == 0 or step == 1:
            print(f"step={step} train_loss={loss_for_log:.4f}")

        if step % 250 == 0:
            eval_losses = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(val_records), 1):
                    batch = collate(val_records[i:i + 1])
                    if batch is not None:
                        eval_losses.append(float(compute_loss(model, batch).item()))
            eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else float("nan")
            print(f"step={step} eval_loss={eval_loss:.4f}")

    if not (checkpoint_dir / "adapter_model.safetensors").exists() and not (checkpoint_dir / "adapter_model.bin").exists():
        print("end save, no best yet")
        save_checkpoint(model, tokenizer, step)