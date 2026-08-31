import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.gemma import run_inference
from utils import compute_metrics, get_blank_positions, parse_sudoku_string, score_prediction, verify_board_validity


def load_jsonl(path, limit=None):
    records = []
    f = open(path)
    for i, line in enumerate(f):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if line:
            records.append(json.loads(line))
    f.close()
    return records


def bucket(diff):
    if diff < 0.3:
        return "easy"
    if diff < 0.7:
        return "medium"
    return "hard"


def evaluate(records):
    out = ROOT / "artifacts" / "results"
    out.mkdir(parents=True, exist_ok=True)
    pred_grids = []
    true_grids = []
    blank_sets = []
    per_bucket = {"easy": [], "medium": [], "hard": []}
    samples_f = open(out / "gemma_eval_samples.jsonl", "w")
    t0 = time.time()
    parsed = 0
    valid = 0

    for i, rec in enumerate(records):
        prompt = rec["prompt_text"]
        true_grid = np.array(rec["grid_solution"], dtype=np.int32)
        blanks = get_blank_positions(prompt)
        diff = float(rec.get("difficulty", 0.0))
        raw = run_inference(str(ROOT / "artifacts" / "checkpoints" / "gemma-sudoku-lora" / "best"), prompt)
        pred = parse_sudoku_string(raw)
        score = score_prediction(pred, true_grid, blanks)

        if pred is not None:
            parsed += 1
            if verify_board_validity(pred):
                valid += 1

        pred_grids.append(pred)
        true_grids.append(true_grid)
        blank_sets.append(blanks)
        per_bucket[bucket(diff)].append(score)
        samples_f.write(json.dumps({
            "id": rec.get("id", f"row_{i}"),
            "difficulty": diff,
            "bucket": bucket(diff),
            "prompt": prompt,
            "raw_output": raw,
            "parsed_ok": pred is not None,
            "valid_board": bool(pred is not None and verify_board_validity(pred)),
            "pct_blank": score["pct_blank"],
            "pct_all": score["pct_all"],
            "correct_blank": score["correct_blank"],
            "total_blank": score["total_blank"],
        }) + "\n")

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[{i + 1}/{len(records)}] parsed={parsed} valid={valid} {rate:.2f} samples/s")

    samples_f.close()
    metrics = compute_metrics(pred_grids, true_grids, blank_sets)
    summary = {
        "model": "google/gemma-4-12B-it",
        "n_samples": len(records),
        "exact_match_acc": metrics["exact_match_acc"],
        "cell_level_acc": metrics["cell_level_acc"],
        "valid_board_acc": metrics["valid_board_acc"],
        "parse_rate": metrics["parse_rate"],
        "pct_all": metrics["pct_all"],
        "pct_blank": metrics["pct_blank"],
        "by_difficulty": {},
    }
    for name, scores in per_bucket.items():
        correct = sum(x["correct_blank"] for x in scores)
        total = sum(x["total_blank"] for x in scores)
        summary["by_difficulty"][name] = {"n": len(scores), "pct_blank": round(correct / total * 100, 2) if total else None}

    f = open(out / "gemma_eval.json", "w")
    json.dump(summary, f, indent=2)
    f.close()
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("SudokuDiffusion - Gemma eval")
    print("device = mps")
    print("checkpoint = artifacts/checkpoints/gemma-sudoku-lora/best")
    print("=" * 60)
    records = load_jsonl(ROOT / "artifacts" / "data" / "val.jsonl", limit=100)
    summary = evaluate(records)
    print("\nFinal results:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:18s} = {v:.4f}")
        else:
            print(f"{k:18s} = {v}")
