import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.llada import generate, load_model_for_inference
from utils import *


def load_val_records(n, max_difficulty=None):
    records = []
    f = open(ROOT / "artifacts" / "data" / "val.jsonl")
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if max_difficulty is not None:
            diff = rec.get("difficulty")
            if diff is None or float(diff) >= max_difficulty:
                continue
        records.append(rec)
        if len(records) == n:
            break
    f.close()
    return records


def evaluate(n_samples=20, num_steps=6, unmask_k=1, max_difficulty=None):
    records = load_val_records(n_samples, max_difficulty)
    print(f"evaluating {len(records)} samples, canvas_len=199, steps={num_steps}, unmask_k={unmask_k}, device=mps")

    model, tokenizer = load_model_for_inference(ROOT / "artifacts" / "checkpoints" / "illada-sudoku-lora")

    all_correct = 0
    all_total = 0
    blank_correct = 0
    blank_total = 0
    sample_results = []

    for idx, rec in enumerate(records):
        prompt = rec["prompt_text"]
        blanks = get_blank_positions(prompt)
        pred_text = generate(model, tokenizer, prompt, num_steps=num_steps, unmask_k=unmask_k)
        pred_grid = parse_sudoku_string(pred_text)
        scores = score_prediction(pred_grid, rec["grid_solution"], blanks)
        all_correct += scores["correct_all"]
        all_total += scores["total_all"]
        blank_correct += scores["correct_blank"]
        blank_total += scores["total_blank"]
        sample_results.append({
            "id": rec.get("id", idx),
            "difficulty": rec.get("difficulty"),
            "prompt_text": prompt,
            "solution_text": rec.get("solution_text", ""),
            "pred_text": pred_text,
            **scores,
        })
        print(f"sample={idx + 1}/{len(records)} pct_all={scores['pct_all']:.1f}% pct_blank={scores['pct_blank'] or 0.0:.1f}%")

    return {
        "n_samples": len(records),
        "num_steps": num_steps,
        "unmask_k": unmask_k,
        "pct_all": round(all_correct / all_total * 100, 2) if all_total else 0.0,
        "pct_blank": round(blank_correct / blank_total * 100, 2) if blank_total else 0.0,
        "pct_blank_by_difficulty": bucket_stats(sample_results),
        "samples": sample_results,
    }


if __name__ == "__main__":
    results = evaluate(
        n_samples=100,
        num_steps=199,
        unmask_k=4,
    )
    out = ROOT / "artifacts" / "eval_llada.json"
    f = open(out, "w")
    json.dump(results, f, indent=2)
    f.close()
    print(f"pct_all: {results['pct_all']:.1f}%")
    print(f"pct_blank: {results['pct_blank']:.1f}%")
    print(f"saved to: {out}")
