import re
import json
import pickle
import asyncio
import faiss
import numpy as np
import csv
import os
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utility.inference import InferenceEngine


# =================================================
# Config
# =================================================
MODELS = [
    "smollm2:135m",
    "qwen2.5:0.5b",
    "qwen2.5:7b",
]

#EMBED_MODEL = "all-MiniLM-L6-v2" 
EMBED_MODEL = "MongoDB/mdbr-leaf-ir"

CHUNK_CONFIGS = {
    # "T256_O64": {
    #     "bm25": "data/bm25_indices/T256_O64/bm25.pkl",
    #     "faiss": "data/vector_indices/T256_O64/faiss.index",
    #     "meta":  "data/vector_indices/T256_O64/metas.pkl",
    # },
    # "T512_O128": {
    #     "bm25": "data/bm25_indices/T512_O128/bm25.pkl",
    #     "faiss": "data/vector_indices/T512_O128/faiss.index",
    #     "meta":  "data/vector_indices/T512_O128/metas.pkl",
    # },
    # "W100": {
    #     "bm25": "data/bm25_indices/W100/bm25.pkl",
    #     "faiss": "data/vector_indices/W100/faiss.index",
    #     "meta":  "data/vector_indices/W100/metas.pkl",
    # },
    "W100": {
        "faiss": "data/mdbr_teacher_vector_indices/W100/faiss.index",
        "meta":  "data/mdbr_teacher_vector_indices/W100/metas.pkl",
    },
}

TOP_K_LIST = [1,2,3]
SAMPLE_SIZE = 10
SEED = 42

OUTPUT_CSV = "leaf_rag_grid_results_2.csv"


# =================================================
# Utils
# =================================================
def normalize(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def parse_answers(ans):
    if ans is None:
        return []
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except Exception:
            return [normalize(ans)]
    flat = []
    for a in ans:
        if isinstance(a, list):
            flat.extend(a)
        else:
            flat.append(a)
    return list({normalize(a) for a in flat if normalize(a)})


def answer_in_text(text, ans):
    if len(ans) < 4:
        return False
    return ans in text


def accuracy(pred, answers):
    pred = normalize(pred)
    return int(any(answer_in_text(pred, a) for a in answers))


def f1_score(pred, answers):
    pred = normalize(pred)
    pred_tokens = set(pred.split())
    if not pred_tokens:
        return 0.0

    for a in answers:
        if answer_in_text(pred, a):
            return 1.0

    best = 0.0
    for a in answers:
        a_tokens = set(a.split())
        if not a_tokens:
            continue
        common = pred_tokens & a_tokens
        if not common:
            continue
        p = len(common) / len(pred_tokens)
        r = len(common) / len(a_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


def hit_at_k(chunks, answers):
    for c in chunks:
        c = normalize(c)
        for a in answers:
            if answer_in_text(c, a):
                return 1
    return 0


def load_done_experiments(csv_path):
    """
    이미 수행된 (model, retrieval, chunk, top_k) 조합 로드
    """
    done = set()
    if not os.path.exists(csv_path):
        return done

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((
                row["model"],
                row["retrieval"],
                row["chunk"],
                int(row["top_k"]),
            ))
    return done


# =================================================
# Main experiment
# =================================================
async def main():
    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    embedder = SentenceTransformer(EMBED_MODEL)

    done_experiments = load_done_experiments(OUTPUT_CSV)
    print(f"[INFO] Loaded {len(done_experiments)} completed experiments")

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "model", "retrieval", "chunk", "top_k",
                "recall@k", "accuracy", "f1",
                "accuracy_given_hit", "hit_and_wrong_ratio",
            ])

        for MODEL in MODELS:
            print(f"\n===== Evaluating MODEL: {MODEL} =====")
            engine_rag = InferenceEngine("NAIVE_RAG", MODEL, verbose=False)

            for chunk_name, paths in CHUNK_CONFIGS.items():
                # bm25_data = pickle.load(open(paths["bm25"], "rb"))
                # bm25 = bm25_data["bm25"]
                # bm25_corpus = [m["text"] for m in bm25_data["metas"]]

                faiss_index = faiss.read_index(paths["faiss"])
                faiss_meta = pickle.load(open(paths["meta"], "rb"))
                vec_corpus = [m["text"] for m in faiss_meta]

                for top_k in TOP_K_LIST:
                    for retrieval in ["BM25", "VEC"]:
                        key = (MODEL, retrieval, chunk_name, top_k)
                        if key in done_experiments:
                            print(f"[SKIP] {key}")
                            continue

                        stats_hit, stats_acc, stats_f1 = [], [], []

                        for ex in tqdm(popqa, desc=f"{MODEL} | {retrieval} | {chunk_name} | K={top_k}"):
                            q = ex["question"]
                            answers = parse_answers(ex.get("possible_answers", []))
                            if not answers:
                                continue

                            if retrieval == "BM25":
                                scores = bm25.get_scores(q.split())
                                idx = np.argsort(scores)[::-1][:top_k]
                                chunks = [bm25_corpus[i] for i in idx]
                            else:
                                q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
                                _, idx = faiss_index.search(q_emb, top_k)
                                chunks = [vec_corpus[i] for i in idx[0] if i != -1]

                            pred = await engine_rag(q, chunks)

                            h = hit_at_k(chunks, answers)
                            a = accuracy(pred, answers)
                            f1 = f1_score(pred, answers)

                            stats_hit.append(h)
                            stats_acc.append(a)
                            stats_f1.append(f1)

                        hit = np.array(stats_hit)
                        acc = np.array(stats_acc)
                        f1 = np.array(stats_f1)

                        writer.writerow([
                            MODEL,
                            retrieval,
                            chunk_name,
                            top_k,
                            hit.mean(),
                            acc.mean(),
                            f1.mean(),
                            acc[hit == 1].mean() if hit.sum() else 0.0,
                            ((hit == 1) & (acc == 0)).mean(),
                        ])

                        done_experiments.add(key)

    print(f"\n[DONE] Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
