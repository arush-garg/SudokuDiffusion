import re

import numpy as np
from transformers import AutoTokenizer


def format_sudoku_string(grid, is_input=True):
    rows = []
    for r in range(9):
        cells = []
        for c in range(9):
            x = int(grid[r][c])
            if is_input and x == 0:
                cells.append(".")
            else:
                cells.append(str(x))
        rows.append("R" + str(r + 1) + ": " + " ".join(cells))

    if is_input:
        return "Input: " + " | ".join(rows)
    return "Solution: " + " | ".join(rows)


def parse_sudoku_string(text):
    if "Solution:" in text:
        text = text.split("Solution:", 1)[1]
    digits = re.findall(r"\b[1-9]\b", text)
    if len(digits) != 81:
        return None
    return np.array([int(x) for x in digits], dtype=np.int32).reshape(9, 9)


def get_blank_positions(prompt_text):
    blanks = set()
    text = prompt_text.replace("Input:", "")
    for r, row in enumerate(text.split("|")[:9]):
        row = re.sub(r"R\d+:", "", row).strip().split()
        for c, val in enumerate(row[:9]):
            if val == ".":
                blanks.add((r, c))
    return blanks


def score_prediction(pred_grid, true_grid, blanks):
    total_blank = len(blanks)
    if pred_grid is None:
        return {
            "correct_all": 0,
            "total_all": 81,
            "pct_all": 0.0,
            "correct_blank": 0,
            "total_blank": total_blank,
            "pct_blank": 0.0 if total_blank else None,
        }

    pred = np.array(pred_grid, dtype=np.int32).reshape(9, 9)
    true = np.array(true_grid, dtype=np.int32).reshape(9, 9)
    match = pred == true
    correct_all = int(match.sum())
    correct_blank = 0
    for r, c in blanks:
        if match[r, c]:
            correct_blank += 1

    return {
        "correct_all": correct_all,
        "total_all": 81,
        "pct_all": round(correct_all / 81 * 100, 2),
        "correct_blank": correct_blank,
        "total_blank": total_blank,
        "pct_blank": round(correct_blank / total_blank * 100, 2) if total_blank else None,
    }


def verify_board_validity(grid):
    grid = np.array(grid, dtype=np.int32)
    if grid.shape != (9, 9):
        return False

    good = set(range(1, 10))
    for i in range(9):
        if set(grid[i].tolist()) != good:
            return False
        if set(grid[:, i].tolist()) != good:
            return False

    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if set(grid[r:r + 3, c:c + 3].flatten().tolist()) != good:
                return False
    return True


def compute_metrics(pred_grids, true_grids, blanks_list=None):
    n = len(true_grids)
    if n == 0:
        out = {"exact_match_acc": 0.0, "cell_level_acc": 0.0, "valid_board_acc": 0.0, "parse_rate": 0.0}
        if blanks_list is not None:
            out["pct_all"] = 0.0
            out["pct_blank"] = 0.0
        return out

    exact = 0
    correct = 0
    valid = 0
    parsed = 0
    blank_correct = 0
    blank_total = 0

    for i, pred in enumerate(pred_grids):
        if pred is None:
            if blanks_list is not None:
                blank_total += len(blanks_list[i])
            continue

        parsed += 1
        score = score_prediction(pred, true_grids[i], blanks_list[i] if blanks_list is not None else [])
        correct += score["correct_all"]
        if score["correct_all"] == 81:
            exact += 1
        if verify_board_validity(pred):
            valid += 1
        if blanks_list is not None:
            blank_correct += score["correct_blank"]
            blank_total += score["total_blank"]

    out = {
        "exact_match_acc": exact / n,
        "cell_level_acc": correct / (n * 81),
        "valid_board_acc": valid / n,
        "parse_rate": parsed / n,
    }
    if blanks_list is not None:
        out["pct_all"] = round(correct / (n * 81) * 100, 2)
        out["pct_blank"] = round(blank_correct / blank_total * 100, 2) if blank_total else 0.0
    return out


def bucket_stats(samples):
    buckets = {"easy": [], "medium": [], "hard": [], "easy_medium": []}
    for sample in samples:
        difficulty = sample.get("difficulty")
        if difficulty is None:
            continue

        if float(difficulty) < 0.3:
            name = "easy"
        elif float(difficulty) < 0.7:
            name = "medium"
        else:
            name = "hard"

        buckets[name].append(sample)
        if name != "hard":
            buckets["easy_medium"].append(sample)

    out = {}
    for name, rows in buckets.items():
        correct_blank = sum(x.get("correct_blank", 0) for x in rows)
        total_blank = sum(x.get("total_blank", 0) for x in rows)
        correct_all = sum(x.get("correct_all", 0) for x in rows)
        total_all = sum(x.get("total_all", 81) for x in rows)
        out[name] = {
            "n": len(rows),
            "pct_blank": round(correct_blank / total_blank * 100, 2) if total_blank else 0.0,
            "pct_all": round(correct_all / total_all * 100, 2) if total_all else 0.0,
        }
    return out


def load_tokenizer(model_id, trust_remote_code=False):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok
