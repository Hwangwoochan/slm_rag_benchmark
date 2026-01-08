import re
import asyncio
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utility.inference import InferenceEngine


MODEL = "smollm2:135m"
MAX_SAMPLES = 100


# ---------------------
# Normalization & F1
# ---------------------
def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def token_f1(pred: str, answers: list[str]) -> float:
    pred_toks = set(normalize(pred).split())
    if not pred_toks:
        return 0.0

    best = 0.0
    for a in answers:
        ans_toks = set(normalize(a).split())
        if not ans_toks:
            continue
        common = pred_toks & ans_toks
        if not common:
            continue
        p = len(common) / len(pred_toks)
        r = len(common) / len(ans_toks)
        best = max(best, 2 * p * r / (p + r))
    return best


# ---------------------
# Main
# ---------------------
async def main():
    engine = InferenceEngine(
        mode="ONLY_SLM",
        model=MODEL,
        verbose=False,
    )

    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.select(range(MAX_SAMPLES))

    f1s = []

    for ex in tqdm(popqa, desc="ONLY_SLM Eval"):
        q = ex["question"]
        answers = ex.get("possible_answers", [])
        if not answers:
            continue

        pred = await engine(question=q)
        f1s.append(token_f1(pred, answers))

    print(f"\nONLY_SLM Avg F1: {np.mean(f1s):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
