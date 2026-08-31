# Sudoku Diffusion

## What I Did

I fine-tuned iLLaDA-8B (a diffusion LM) and Gemma-4-12B to solve Sudoku puzzles, then benchmarked both against shot baselines and Llama 3.3 70B across easy, medium and hard puzzles.

The results are in the table below, and they clearly show that diffusion models are better suited to Sudoku than autoregressive models. 


| Difficulty | iLLaDA zero-shot | Gemma-4 zero-shot | iLLaDA fine-tuned | Gemma-4 fine-tuned | Llama 3.3 70B |
|------------|------------------|-------------------|-------------------|--------------------|---------------|
| easy       | ~35.8%           | 72.5%             | 92.7%             | **95.7%**          | 35.2%         |
| medium     | ~8.2%            | 21.5%             | **69.6%**         | 38.9%              | 14.4%         |
| hard       | ~0.0%            | 3.6%              | **27.4%**         | 8.9%               | 7.5%          |

<i>Metric: % of originally-blank cells that were predicted correctly by the model. All models evaluated on the same 100 held-out validation puzzles.</i>

## What was Difficult

I haven't really done much with diffusion models before, so learning how they work and how to fine-tune them was a bit of a challenge. I also had to iterate with the data formatting and model output parsing to get everything working correctly (like ensuring every cell maps to exactly one token).

## How it works

Every puzzle is formatted so each cell maps to exactly one token:

```
Input: R1: 5 3 . . 7 . . . . | R2: ... | R9: . . . . 8 . . 7 9
Solution: R1: 5 3 4 6 7 8 9 1 2 | R2: ... | R9: 3 4 5 2 8 6 1 7 9
```

## Setup

```bash
pip install -r requirements.txt
python generate_data.py       # regenerate datasets into artifacts/data/
```

You can access the best iLLaDA LoRA adapter on Hugging Face: [Techno03/illada-8b-sudoku-lora](https://huggingface.co/Techno03/illada-8b-sudoku-lora).