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

# LEAF Student 모델 (쿼리 인코더)
#EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"


EMBED_MODEL = "MongoDB/mdbr-leaf-ir"


# Arctic v1.5 모델은 쿼리 시 아래 문구가 필수입니다.
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_CONFIGS = {
    # 새로 생성한 T512_O128 인덱스 경로 반영
#    "T512_O128": {
#        "faiss": "data/mdbr_teacher_vector_indices/T512_O128/faiss.index",
#        "meta":  "data/mdbr_teacher_vector_indices/T512_O128/metas.pkl",
#    },
#    비교를 위해 기존 W100도 유지하고 싶다면 아래 주석을 해제하세요.
    "W100": {
        "faiss": "data/mdbr_teacher_vector_indices/W100/faiss.index",
        "meta":  "data/mdbr_teacher_vector_indices/W100/metas.pkl",
    },
}

TOP_K_LIST = [1, 2, 3]
SAMPLE_SIZE = 500  # 성능 확인 후 숫자를 늘려보세요
SEED = 42

OUTPUT_CSV = "leaf_rag_grid_results_final.csv"


# =================================================
# Utils
# =================================================
def normalize(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()

def parse_answers(ans):
    if ans is None: return []
    if isinstance(ans, str):
        try: ans = json.loads(ans)
        except: return [normalize(ans)]
    flat = []
    for a in ans:
        if isinstance(a, list): flat.extend(a)
        else: flat.append(a)
    return list({normalize(a) for a in flat if normalize(a)})

def answer_in_text(text, ans):
    # 너무 짧은 답변(예: "the", "a")은 무시하거나 길이를 조정하세요.
    if len(ans) < 2: return False 
    return ans in text

def accuracy(pred, answers):
    pred = normalize(pred)
    return int(any(answer_in_text(pred, a) for a in answers))

def f1_score(pred, answers):
    pred = normalize(pred)
    pred_tokens = set(pred.split())
    if not pred_tokens: return 0.0
    for a in answers:
        if answer_in_text(pred, a): return 1.0
    best = 0.0
    for a in answers:
        a_tokens = set(a.split())
        if not a_tokens: continue
        common = pred_tokens & a_tokens
        if not common: continue
        p = len(common) / len(pred_tokens)
        r = len(common) / len(a_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best

def hit_at_k(chunks, answers):
    for c in chunks:
        c = normalize(c)
        for a in answers:
            if answer_in_text(c, a): return 1
    return 0

def load_done_experiments(csv_path):
    done = set()
    if not os.path.exists(csv_path): return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["model"], row["retrieval"], row["chunk"], int(row["top_k"])))
    return done


# =================================================
# Main experiment
# =================================================
async def main():
    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    # LEAF Student 모델 로드
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    done_experiments = load_done_experiments(OUTPUT_CSV)
    print(f"[INFO] Loaded {len(done_experiments)} completed experiments")

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "model", "retrieval", "chunk", "top_k",
                "recall@k", "accuracy", "f1",
                "accuracy_given_hit", "hit_and_wrong_ratio",
            ])

        for MODEL in MODELS:
            print(f"\n===== Evaluating MODEL: {MODEL} =====")
            # InferenceEngine은 사용자가 정의한 유틸리티를 사용한다고 가정합니다.
            engine_rag = InferenceEngine("NAIVE_RAG", MODEL, verbose=False)

            for chunk_name, paths in CHUNK_CONFIGS.items():
                if not os.path.exists(paths["faiss"]):
                    print(f"[WARN] Index not found: {paths['faiss']}")
                    continue

                # 인덱스 로드
                faiss_index = faiss.read_index(paths["faiss"])
                faiss_meta = pickle.load(open(paths["meta"], "rb"))
                vec_corpus = [m["text"] for m in faiss_meta]

                for top_k in TOP_K_LIST:
                    for retrieval in ["VEC"]:
                        key = (MODEL, retrieval, chunk_name, top_k)
                        if key in done_experiments:
                            print(f"[SKIP] {key}")
                            continue

                        stats_hit, stats_acc, stats_f1 = [], [], []

                        for ex in tqdm(popqa, desc=f"{MODEL} | {retrieval} | {chunk_name} | K={top_k}"):
                            q = ex["question"]
                            answers = parse_answers(ex.get("possible_answers", []))
                            if not answers: continue

                            # ----------------------------------------------------
                            # 핵심 수정: Arctic v1.5 전용 쿼리 인코딩 (Prefix 추가)
                            # ----------------------------------------------------
                            # 쿼리 임베딩 생성 시 Prefix 필수
                            q_with_instruction = ARCTIC_QUERY_PREFIX + q
                            q_emb = embedder.encode(
                                q_with_instruction, 
                                normalize_embeddings=True, 
                                convert_to_numpy=True
                            ).astype("float32")
                            
                            # FAISS 검색을 위한 차원 조정 (1, Dim)
                            if q_emb.ndim == 1:
                                q_emb = np.expand_dims(q_emb, axis=0)

                            _, idx = faiss_index.search(q_emb, top_k)
                            chunks = [vec_corpus[i] for i in idx[0] if i != -1]
                            # ----------------------------------------------------

                            # LLM 추론
                            pred = await engine_rag(q, chunks)

                            # 지표 계산
                            h = hit_at_k(chunks, answers)
                            a = accuracy(pred, answers)
                            f1 = f1_score(pred, answers)

                            stats_hit.append(h)
                            stats_acc.append(a)
                            stats_f1.append(f1)

                        # 통계 계산 및 기록
                        hit = np.array(stats_hit)
                        acc = np.array(stats_acc)
                        f1_arr = np.array(stats_f1)

                        writer.writerow([
                            MODEL,
                            retrieval,
                            chunk_name,
                            top_k,
                            hit.mean() if len(hit) > 0 else 0,
                            acc.mean() if len(acc) > 0 else 0,
                            f1_arr.mean() if len(f1_arr) > 0 else 0,
                            acc[hit == 1].mean() if hit.sum() > 0 else 0.0,
                            ((hit == 1) & (acc == 0)).mean() if len(hit) > 0 else 0,
                        ])
                        f.flush() # 실시간 저장
                        done_experiments.add(key)

    print(f"\n[DONE] Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())