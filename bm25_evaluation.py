# bm25_evaluation.py

import re
import pickle
import numpy as np
import asyncio
from datasets import load_dataset
from tqdm import tqdm

from utility.inference import InferenceEngine


# ---------------------
# Config
# ---------------------
BM25_PATH = "data/bm25_indices/T256_O64/bm25.pkl"
TOP_K = 1
MODEL = "smollm2:135m"
MAX_SAMPLES = 10


# ---------------------
# Normalization & Metrics
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


def hit_at_k(chunks: list[str], answers: list[str]) -> int:
    """Recall@K (Hit@K): 정답이 retrieved chunk 안에 존재하는지"""
    chunks_norm = [normalize(c) for c in chunks]
    answers_norm = [normalize(a) for a in answers]

    for c in chunks_norm:
        for a in answers_norm:
            if a in c:
                return 1
    return 0


# ---------------------
# Main
# ---------------------
async def main():
    # Load BM25
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    corpus = [m["text"] for m in data["metas"]]

    # Inference engine (NAIVE_RAG)
    engine = InferenceEngine(
        mode="NAIVE_RAG",
        model=MODEL,
        verbose=False,
    )

    # Dataset
    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.select(range(MAX_SAMPLES))

    hits = []
    f1s = []

    for ex in tqdm(popqa, desc="BM25-RAG Eval"):
        q = ex["question"]
        answers = ex.get("possible_answers", [])
        if not answers:
            continue

        # ---------- Retrieval ----------
        scores = bm25.get_scores(q.split())
        idx = np.argsort(scores)[::-1][:TOP_K]
        chunks = [corpus[i] for i in idx]

        # ---------- Retrieval Evaluation ----------
        hit = hit_at_k(chunks, answers)
        hits.append(hit)

        # ---------- Inference ----------
        pred = await engine(
            question=q,
            retrieved_chunks=chunks,
        )

        # ---------- Generation Evaluation ----------
        f1 = token_f1(pred, answers)
        f1s.append(f1)

        # Debug print (optional)
        print("=" * 80)
        print(f"Q: {q}")
        print(f"Retrieved HIT@{TOP_K}: {hit}")
        print("Prediction:", pred)
        print("Answers:", answers)
        print(f"F1: {f1:.3f}")

    # ---------------------
    # Final Results
    # ---------------------
    hits = np.array(hits)
    f1s = np.array(f1s)

    print("\n================ Final Results ================")
    print(f"BM25 Recall@{TOP_K}: {hits.mean():.4f}")
    print(f"BM25-RAG Avg F1 (@{TOP_K}): {f1s.mean():.4f}")

    # Conditional generation performance
    if hits.sum() > 0:
        print(
            f"F1 | Recall@{TOP_K}=1 only: "
            f"{f1s[hits == 1].mean():.4f}"
        )
    else:
        print(f"F1 | Recall@{TOP_K}=1 only: N/A")


if __name__ == "__main__":
    asyncio.run(main())
