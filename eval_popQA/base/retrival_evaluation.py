import time
import re
import json
import pickle
import csv
import gc
import os

import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# =================================================
# 1. 설정
# =================================================
TARGET_CONFIGS = [
    ("VEC",  "T512_O128", 3),
    ("BM25", "T512_O128", 3),
    ("VEC",  "T256_O64",  3),
    ("BM25", "T256_O64",  3),
    ("VEC",  "T512_O128", 2),
    ("BM25", "T512_O128", 2),
    ("VEC",  "W100",      2),
    ("BM25", "T256_O64",  2),
    ("VEC",  "T256_O64",  2),
    ("BM25", "W100",      3),
]

EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = "./data"
OUTPUT_CSV = "retrieval_benchmark_with_recall.csv"

# 평가 데이터(질문/정답) 설정
DATASET_NAME = "akariasai/PopQA"
DATASET_SPLIT = "test"
SAMPLE_SIZE = 200          # 질문 수
SEED = 42                  # 샘플링 고정(재현성)

# =================================================
# 2. 정답 파싱/리콜 계산 유틸
# =================================================
def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()

def parse_answers(ans):
    """PopQA possible_answers는 list/중첩list/str(json) 등 다양할 수 있어 안전하게 펼침."""
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

    out = []
    for a in flat:
        a_n = normalize(a)
        if a_n:
            out.append(a_n)
    # 중복 제거
    return list(set(out))

def hit_at_k_texts(texts, answers_norm):
    """Top-k retrieved 텍스트들 중 하나라도 정답 문자열을 포함하면 1 else 0"""
    if not answers_norm:
        return 0
    for t in texts:
        t_n = normalize(t)
        for a in answers_norm:
            # 너무 짧은 답은 오탐 많아서 무시 (원하면 기준 변경)
            if len(a) < 2:
                continue
            if a in t_n:
                return 1
    return 0

# =================================================
# 3. 메인 실행
# =================================================
def run_retrieval_benchmarks():
    print(f"--- Retrieval 벤치마크(질문 샘플링 + recall@k 계산) 시작 ---")

    # 1) 평가 데이터에서 질문/정답 샘플링 (SEED 고정)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    test_items = []
    for ex in ds:
        q = ex.get("question", "")
        answers = parse_answers(ex.get("possible_answers", []))
        if q and answers:
            test_items.append((q, answers))

    print(f"[INFO] Loaded {len(test_items)} Q/A items from {DATASET_NAME}:{DATASET_SPLIT}")

    # 2) 임베딩 모델 1회 로드
    print(f"[INFO] Loading embed model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # 3) CSV 헤더
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Retrieval_Type", "Chunk_Name", "Top_K",
            "Recall@K(hit@k)", "Avg_Time(s)", "Min_Time(s)", "Max_Time(s)"
        ])

    # 4) 조합 루프
    for ret_type, chunk, top_k in TARGET_CONFIGS:
        print(f"\n🔍 [측정] {ret_type} | {chunk} | K={top_k}")

        # 이전 루프 메모리 정리
        index = None
        data_obj = None
        gc.collect()

        # 인덱스/메타 로드
        try:
            if ret_type == "VEC":
                idx_path = f"{DATA_DIR}/vector_indices/{chunk}/faiss.index"
                meta_path = f"{DATA_DIR}/vector_indices/{chunk}/metas.pkl"

                index = faiss.read_index(idx_path)
                with open(meta_path, "rb") as f:
                    data_obj = pickle.load(f)  # list[{"text":...}, ...] 형태라고 가정
            else:
                bm25_path = f"{DATA_DIR}/bm25_indices/{chunk}/bm25.pkl"
                with open(bm25_path, "rb") as f:
                    data_obj = pickle.load(f)
                    index = data_obj["bm25"]    # bm25 객체
        except Exception as e:
            print(f"    >> 로드 실패: {e}")
            continue

        r_times = []
        hits = []

        # 질문 루프
        for q, answers_norm in tqdm(test_items, desc="      측정 중", leave=False):
            start_r = time.perf_counter()

            if ret_type == "VEC":
                q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")
                _, idx = index.search(q_emb, top_k)

                # 텍스트 리스트를 새로 만들되(리콜 계산 위해 필요),
                # top_k만큼만 뽑으니 부담이 적음
                texts = []
                for i in idx[0]:
                    if i != -1:
                        texts.append(data_obj[i]["text"])
            else:
                scores = index.get_scores(q.split())
                idx = np.argsort(scores)[::-1][:top_k]
                texts = [data_obj["metas"][i]["text"] for i in idx]

            r_times.append(time.perf_counter() - start_r)
            hits.append(hit_at_k_texts(texts, answers_norm))

        # 결과 집계
        recall_k = float(np.mean(hits)) if hits else 0.0
        avg_r = float(np.mean(r_times)) if r_times else 0.0
        min_r = float(np.min(r_times)) if r_times else 0.0
        max_r = float(np.max(r_times)) if r_times else 0.0

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ret_type, chunk, top_k,
                f"{recall_k:.4f}",
                f"{avg_r:.6f}", f"{min_r:.6f}", f"{max_r:.6f}",
            ])

        print(f"    >> 완료: Recall@{top_k}={recall_k:.4f} | Avg {avg_r:.6f}s")

    print(f"\n[완료] 결과 저장: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_retrieval_benchmarks()
