import re
import pickle
import asyncio
import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utility.inference import InferenceEngine


FAISS_INDEX_PATH = "data/vector_indices/T256_O64/faiss.index"
META_PATH = "data/vector_indices/T256_O64/metas.pkl"

MODEL = "smollm2:135m"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
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
    # Load FAISS + metas
    index = faiss.read_index(FAISS_INDEX_PATH)
    metas = pickle.load(open(META_PATH, "rb"))

    corpus = [m["text"] for m in metas]

    # Embedder (query side)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Inference engine (RAG prompt)
    engine = InferenceEngine(
        mode="NAIVE_RAG",
        model=MODEL,
        verbose=False,
    )

    # Dataset
    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.select(range(MAX_SAMPLES))

    f1s = []
    skipped = 0

    for ex in tqdm(popqa, desc="Vector-RAG Eval"):
        q = ex["question"]
        answers = ex.get("possible_answers", [])
        if not answers:
            skipped += 1
            continue

        # Query embedding
        q_emb = embedder.encode([q], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        # Vector retrieval
        _, idx = index.search(q_emb, TOP_K)
        chunks = [corpus[i] for i in idx[0] if i != -1]

        # Inference (!!! 핵심 수정: context= 쓰지 말고 retrieved_chunks=로)
        pred = await engine(question=q, retrieved_chunks=chunks)

        # Eval
        f1s.append(token_f1(pred, answers))

    avg_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"\nEvaluated: {len(popqa) - skipped} / {len(popqa)} (skipped {skipped})")
    print(f"Vector-RAG Avg F1 (@{TOP_K}): {avg_f1:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
