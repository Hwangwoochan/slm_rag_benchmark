# rag_comparison_popqa.py

import re
import json
import pickle
import asyncio
import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utility.inference import InferenceEngine


# =================================================
# Config
# =================================================
BM25_PATH = "data/bm25_indices/T256_O64/bm25.pkl"
FAISS_INDEX_PATH = "data/vector_indices/T256_O64/faiss.index"
FAISS_META_PATH = "data/vector_indices/T256_O64/metas.pkl"

MODEL = "smollm2:135m"
EMBED_MODEL = "all-MiniLM-L6-v2"

TOP_K_BM25 = 1
TOP_K_VEC = 1
MAX_SAMPLES = 100

PRINT_WRONG_ONLY = True
PRINT_CHUNKS = False


# =================================================
# Text utils
# =================================================
def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def parse_answers(possible_answers):
    if possible_answers is None:
        return []

    if isinstance(possible_answers, str):
        s = possible_answers.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                possible_answers = json.loads(s)
            except Exception:
                return [normalize(s)]

    flat = []
    if isinstance(possible_answers, list):
        for a in possible_answers:
            if isinstance(a, list):
                flat.extend(a)
            else:
                flat.append(a)
    else:
        flat = [possible_answers]

    out, seen = [], set()
    for a in flat:
        a_norm = normalize(a)
        if a_norm and a_norm not in seen:
            seen.add(a_norm)
            out.append(a_norm)
    return out


# =================================================
# Metrics (PopQA-style)
# =================================================
def answer_in_text(text_norm: str, ans_norm: str) -> bool:
    if len(ans_norm) < 4:
        return False
    return ans_norm in text_norm


def accuracy_em(pred: str, answers_norm: list[str]) -> int:
    pred_norm = normalize(pred)
    return int(any(answer_in_text(pred_norm, a) for a in answers_norm))


def token_f1(pred: str, answers_norm: list[str]) -> float:
    pred_norm = normalize(pred)
    pred_tokens = set(pred_norm.split())
    if not pred_tokens:
        return 0.0

    for a in answers_norm:
        if answer_in_text(pred_norm, a):
            return 1.0

    best = 0.0
    for a in answers_norm:
        ans_tokens = set(a.split())
        if not ans_tokens:
            continue
        common = pred_tokens & ans_tokens
        if not common:
            continue
        p = len(common) / len(pred_tokens)
        r = len(common) / len(ans_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


def hit_at_k(chunks: list[str], answers_norm: list[str]) -> int:
    for c in chunks:
        c_norm = normalize(c)
        for a in answers_norm:
            if answer_in_text(c_norm, a):
                return 1
    return 0


# =================================================
# Main
# =================================================
async def main():
    # ---------- Load BM25 ----------
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    bm25_corpus = [m["text"] for m in bm25_data["metas"]]

    # ---------- Load FAISS ----------
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    faiss_metas = pickle.load(open(FAISS_META_PATH, "rb"))
    vec_corpus = [m["text"] for m in faiss_metas]
    embedder = SentenceTransformer(EMBED_MODEL)

    # ---------- Engines ----------
    engine_only = InferenceEngine("ONLY_SLM", MODEL, verbose=False)
    engine_rag = InferenceEngine("NAIVE_RAG", MODEL, verbose=False)

    # ---------- Dataset ----------
    #popqa = load_dataset("akariasai/PopQA", split="test").select(range(MAX_SAMPLES))

    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.shuffle(seed=42).select(range(MAX_SAMPLES))
    
    # ---------- Metrics ----------
    stats = {
        "bm25_hit": [],
        "vec_hit": [],
        "only_acc": [],
        "bm25_acc": [],
        "vec_acc": [],
        "only_f1": [],
        "bm25_f1": [],
        "vec_f1": [],
    }

    for ex in tqdm(popqa, desc="ONLY vs BM25 vs Vector RAG"):
        q = ex["question"]
        answers = parse_answers(ex.get("possible_answers", []))
        if not answers:
            continue

        # ===== ONLY SLM =====
        pred_only = await engine_only(question=q)
        stats["only_acc"].append(accuracy_em(pred_only, answers))
        stats["only_f1"].append(token_f1(pred_only, answers))

        # ===== BM25 =====
        scores = bm25.get_scores(q.split())
        idx = np.argsort(scores)[::-1][:TOP_K_BM25]
        bm25_chunks = [bm25_corpus[i] for i in idx]
        stats["bm25_hit"].append(hit_at_k(bm25_chunks, answers))

        pred_bm25 = await engine_rag(question=q, retrieved_chunks=bm25_chunks)
        stats["bm25_acc"].append(accuracy_em(pred_bm25, answers))
        stats["bm25_f1"].append(token_f1(pred_bm25, answers))

        # ===== Vector =====
        q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
        _, idx = faiss_index.search(q_emb, TOP_K_VEC)
        vec_chunks = [vec_corpus[i] for i in idx[0] if i != -1]
        stats["vec_hit"].append(hit_at_k(vec_chunks, answers))

        pred_vec = await engine_rag(question=q, retrieved_chunks=vec_chunks)
        stats["vec_acc"].append(accuracy_em(pred_vec, answers))
        stats["vec_f1"].append(token_f1(pred_vec, answers))

        # ----- Debug -----
  # ----- Debug: per-sample detailed output -----
        print("=" * 110)
        print(f"Q: {q}")
        print(f"ANS: {answers}")

        print(f"\n[ONLY_SLM]")
        print(f"  ACC: {stats['only_acc'][-1]} | F1: {stats['only_f1'][-1]:.3f}")
        print(f"  PRED: {pred_only}")

        print(f"\n[BM25 + SLM]")
        print(f"  HIT@{TOP_K_BM25}: {stats['bm25_hit'][-1]}")
        print(f"  ACC: {stats['bm25_acc'][-1]} | F1: {stats['bm25_f1'][-1]:.3f}")
        print(f"  PRED: {pred_bm25}")

        print(f"\n[Vector + SLM]")
        print(f"  HIT@{TOP_K_VEC}: {stats['vec_hit'][-1]}")
        print(f"  ACC: {stats['vec_acc'][-1]} | F1: {stats['vec_f1'][-1]:.3f}")
        print(f"  PRED: {pred_vec}")

        if PRINT_CHUNKS:
            print("\n[BM25 Chunks]")
            for i, c in enumerate(bm25_chunks):
                print(f"  [{i}] {c[:300]}{'...' if len(c) > 300 else ''}")

            print("\n[Vector Chunks]")
            for i, c in enumerate(vec_chunks):
                print(f"  [{i}] {c[:300]}{'...' if len(c) > 300 else ''}")


    # ---------- Summary ----------
    print("\n================ Final Results ================")
    print(f"Samples: {len(stats['only_acc'])}")

    print("\n[Retrieval]")
    print(f"BM25 Recall@{TOP_K_BM25}: {np.mean(stats['bm25_hit']):.4f}")
    print(f"Vector Recall@{TOP_K_VEC}: {np.mean(stats['vec_hit']):.4f}")

    print("\n[Generation Accuracy]")
    print(f"ONLY_SLM Accuracy: {np.mean(stats['only_acc']):.4f}")
    print(f"BM25+SLM Accuracy: {np.mean(stats['bm25_acc']):.4f}")
    print(f"Vector+SLM Accuracy: {np.mean(stats['vec_acc']):.4f}")

    print("\n[Generation F1]")
    print(f"ONLY_SLM F1: {np.mean(stats['only_f1']):.4f}")
    print(f"BM25+SLM F1: {np.mean(stats['bm25_f1']):.4f}")
    print(f"Vector+SLM F1: {np.mean(stats['vec_f1']):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
