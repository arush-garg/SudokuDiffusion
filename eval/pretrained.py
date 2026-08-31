import json
import sys
import time
from pathlib import Path
import openai
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import *


def load_val_records(n):
    records = []
    f = open(ROOT / "artifacts" / "data" / "val.jsonl")
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
        if len(records) == n:
            break
    f.close()
    if not records:
        raise ValueError("no val records")
    return records


def score_sample(pred_text, true_grid, blanks):
    parsed_grid = parse_sudoku_string(pred_text)
    if parsed_grid is None:
        return {
            "exact_match": False,
            "cell_acc": 0.0,
            "valid_board": False,
            "parsed": False,
            "pct_all": 0.0,
            "pct_blank": 0.0,
        }

    pred_arr = np.array(parsed_grid, dtype=np.int32)
    true_arr = np.array(true_grid, dtype=np.int32)
    correct_cells = int(np.sum(pred_arr == true_arr))
    scores = score_prediction(parsed_grid, true_grid, blanks)
    return {
        "exact_match": correct_cells == 81,
        "cell_acc": correct_cells / 81.0,
        "valid_board": verify_board_validity(pred_arr),
        "parsed": True,
        "pct_all": float(scores["pct_all"]),
        "pct_blank": float(scores["pct_blank"] or 0.0),
    }



SYSTEM_PROMPT = (
    "You are an expert Sudoku solver. Given a Sudoku puzzle in the specified "
    "format, output ONLY the solution in the exact same format. Do not include "
    "any explanation.\n\n"
    "The format uses row markers R1 through R9, with cells space-separated and "
    "rows separated by \" | \".\n\n"
    "Example input:  Input: R1: 5 3 . . 7 . . . . | R2: 6 . . 1 9 5 . . . | ...\n"
    "Example output: Solution: R1: 5 3 4 6 7 8 9 1 2 | R2: 6 7 2 1 9 5 3 4 8 | ...\n\n"
    "Output ONLY the Solution line. Nothing else."
)


def call_llm(prompt_text, max_tokens=512):
    client = openai.OpenAI(api_key="not needed", base_url="http://localhost:20128/v1")
    response = client.chat.completions.create(
        model="groq/llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    )
    return (response.choices[0].message.content or "").strip()

def evaluate_model(samples):
    per_sample = []
    pred_grids = []
    true_grids = []
    blank_sets = []
    bucket_items = []
    print(f"\n=== groq/llama-3.3-70b-versatile ({len(samples)} samples) ===")

    total_latency = 0.0
    for i, rec in enumerate(samples, start=1):
        prompt = rec["prompt_text"]
        true_grid = np.array(rec["grid_solution"], dtype=np.int32)
        difficulty = float(rec.get("difficulty", 0.5))
        blanks = get_blank_positions(prompt)
        blank_sets.append(blanks)

        start = time.perf_counter()
        pred_text = call_llm(prompt, max_tokens=512)
        latency = time.perf_counter() - start
        total_latency += latency

        scores = score_sample(pred_text, true_grid, blanks)
        pred_grid = parse_sudoku_string(pred_text)
        pred_grids.append(pred_grid)
        true_grids.append(true_grid)

        per_sample.append({
            "id": rec["id"],
            "model": "groq/llama-3.3-70b-versatile",
            "difficulty": difficulty,
            "pred_text": pred_text,
            "exact_match": scores["exact_match"],
            "cell_acc": round(scores["cell_acc"], 4),
            "valid_board": scores["valid_board"],
            "parsed": scores["parsed"],
            "pct_all": scores["pct_all"],
            "pct_blank": scores["pct_blank"],
            "latency_s": round(latency, 4),
        })

        if scores["parsed"]:
            sp = score_prediction(pred_grid, true_grid, blanks)
            bucket_items.append({
                "difficulty": difficulty,
                "correct_blank": sp["correct_blank"],
                "total_blank": sp["total_blank"],
                "correct_all": sp["correct_all"],
                "total_all": sp["total_all"],
            })
        else:
            bucket_items.append({
                "difficulty": difficulty,
                "correct_blank": 0,
                "total_blank": len(blanks),
                "correct_all": 0,
                "total_all": 81,
            })

        if i % 10 == 0 or i == len(samples):
            avg_lat = total_latency / i
            print(
                f"[{i:3d}/{len(samples)}] "
                f"parse={sum(s['parsed'] for s in per_sample) / i:.3f} "
                f"exact={sum(s['exact_match'] for s in per_sample) / i:.3f} "
                f"pct_all={sum(s['pct_all'] for s in per_sample) / i:.1f} "
                f"pct_blank={sum(s['pct_blank'] for s in per_sample) / i:.1f} "
                f"avg_lat={avg_lat:.2f}s"
            )

    metrics = compute_metrics(pred_grids, true_grids, blank_sets)
    return {
        "model": "groq/llama-3.3-70b-versatile",
        "n_samples": len(samples),
        "exact_match_acc": metrics["exact_match_acc"],
        "cell_level_acc": metrics["cell_level_acc"],
        "valid_board_acc": metrics["valid_board_acc"],
        "parse_rate": metrics["parse_rate"],
        "avg_latency_s": round(total_latency / max(len(samples), 1), 4),
        "pct_all": metrics["pct_all"],
        "pct_blank": metrics["pct_blank"],
        "difficulty_buckets": bucket_stats(bucket_items),
    }, per_sample


def save_results(report, samples):
    out = ROOT / "artifacts" / "results"
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "groq_eval.json", "w")
    json.dump({"llama-3.3-70b": report}, f, indent=2)
    f.close()
    f = open(out / "groq_eval_samples.jsonl", "w")
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
    f.close()
    return out / "groq_eval.json", out / "groq_eval_samples.jsonl"


if __name__ == "__main__":
    print("pretrained eval, n=100, max_tokens=512")
    samples = load_val_records(100)
    report, per_sample = evaluate_model(samples)
    report_path, samples_path = save_results(report, per_sample)

    print("\n" + "=" * 78)
    print("Groq Sudoku Eval - Summary")
    print("=" * 78)
    print(f"{report['model']:<32} n={report['n_samples']} pct_all={report['pct_all']:.2f}% pct_blank={report['pct_blank']:.2f}%")
    print(f"exact={report['exact_match_acc']:.3f} valid={report['valid_board_acc']:.3f} parse={report['parse_rate']:.3f} avg_lat={report['avg_latency_s']:.2f}s")
    print(f"\n{'bucket':<12} {'n':>4} {'pct_all':>9} {'pct_blank':>10}")
    for bucket, b in report["difficulty_buckets"].items():
        print(f"{bucket:<12} {b['n']:>4} {b['pct_all']:>8.2f}% {b['pct_blank']:>9.2f}%")
    print(f"saved aggregate: {report_path}")
    print(f"saved samples: {samples_path}")
