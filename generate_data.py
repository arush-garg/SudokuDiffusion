import json
import multiprocessing as mp
import random
from pathlib import Path
import numpy as np
from sudoku import Sudoku

from utils import format_sudoku_string


ROOT = Path(__file__).resolve().parent


def build_record(puzzle, solution, difficulty, idx):
    return {
        "id": f"sudoku_{idx:05d}",
        "difficulty": round(difficulty, 4),
        "prompt_text": format_sudoku_string(puzzle, is_input=True),
        "solution_text": format_sudoku_string(solution, is_input=False),
        "grid_solution": solution.tolist(),
    }

def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.close()
    print(f"wrote {len(records):,} records to {path}")


def worker_init(seed):
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    random.seed(seed + worker_id * 1000003)
    np.random.seed(seed + worker_id * 1000003)


def generate_one(args):
    diff, idx = args
    puzzle = Sudoku(3, seed=random.randint(0, 10**9)).difficulty(diff)
    solution = puzzle.solve()
    puzzle_grid = np.array([[cell or 0 for cell in row] for row in puzzle.board], dtype=np.int32)
    solution_grid = np.array([[cell or 0 for cell in row] for row in solution.board], dtype=np.int32)
    return build_record(puzzle_grid, solution_grid, diff, idx)


def generate_split(n, start_idx, seed):
    rng = random.Random(seed)
    args = list(zip([rng.uniform(0.2, 0.8) for _ in range(n)], range(start_idx, start_idx + n)))
    records = []
    with mp.Pool(min(8, mp.cpu_count()), initializer=worker_init, initargs=(seed,)) as pool:
        for i, rec in enumerate(pool.imap(generate_one, args, chunksize=max(1, n // 32))):
            records.append(rec)
            if (i + 1) % 500 == 0:
                print(f"{i + 1:,}/{n:,} ...")
    return records


if __name__ == "__main__":
    print("=" * 60)
    print("SudokuDiffusion - Data Preparation")
    print("=" * 60)

    out = ROOT / "artifacts" / "data"

    print("\n[1/3] Generating train split (50,000) ...")
    train = generate_split(50000, 0, 0)
    write_jsonl(train, out / "train.jsonl")

    print("\n[2/3] Generating val split (2,000) ...")
    val = generate_split(2000, 50000, 1)
    write_jsonl(val, out / "val.jsonl")

    print("\n[3/3] Generating test split (1,000) ...")
    test = generate_split(1000, 52000, 2)
    write_jsonl(test, out / "test.jsonl")
