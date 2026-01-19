import re
import json
import pickle
import asyncio
import faiss
import numpy as np
import csv
import os
import random
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utility.inference import InferenceEngine

# =================================================
# 1. Config (설정)
# =================================================
MODELS = ["smollm2:135m", "qwen2.5:0.5b", "qwen2.5:7b"]
EMBED_MODEL = "MongoDB/mdbr-leaf-ir"

CHUNK_CONFIGS = {
    # "T256_O64": {
    #     "bm25": "data/bm25_indices_techqa/T256_O64/bm25.pkl",
    #     "faiss": "data/minilm_vector_indices_combined/T256_O64/faiss_combined.index",
    #     "meta":  "data/minilm_vector_indices_combined/T256_O64/metas_combined.pkl",
    # },
     "T512_O128": {
         "bm25": "data/bm25_indices_techqa/T512_O128/bm25.pkl",
         "faiss": "data/leaf_minilm_vector_indices_combined/T512_O128/faiss_combined.index",
         "meta":  "data/leaf_minilm_vector_indices_combined/T512_O128/metas_combined.pkl", 
    },
    "W100": {
        "bm25": "data/bm25_indices_techqa/W100/bm25.pkl",
        "faiss": "data/leaf_minilm_vector_indices_combined/W100/faiss_combined.index",
        "meta":  "data/leaf_minilm_vector_indices_combined/W100/metas_combined.pkl",
    }
}

TOP_K_LIST = [2, 3]
SAMPLE_SIZE = 300
SEED = 42

DATA_PATH = "./data/combined_data.jsonl"
OUTPUT_CSV = "rag_techqa_grid_stats_new_leaf.csv"
OUTPUT_JSONL = "rag_techqa_llm_judge_data_leaf.jsonl"

# =================================================
# 2. Utils
# =================================================
def get_processed_keys(file_path):
    """이미 완료된 작업의 키를 추출 (모델, 검색, 청크, K, ID)"""
    processed = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    m = data["metadata"]
                    # 고유 키 생성
                    key = (m["model"], m["retrieval"], m["chunk_config"], m["top_k"], m["id"])
                    processed.add(key)
                except: continue
    return processed

def extract_gold_standard_b(ex):
    relevant_keys = ex.get("all_relevant_sentence_keys", [])
    doc_sentences = ex.get("documents_sentences", [])
    if not relevant_keys or not doc_sentences:
        return ex.get("response", "No gold standard available.")
    gold_facts = []
    for key in relevant_keys:
        try:
            doc_idx_match = re.search(r'\d+', key)
            if not doc_idx_match: continue
            doc_idx = int(doc_idx_match.group())
            if doc_idx < len(doc_sentences):
                for s_id, s_text in doc_sentences[doc_idx]:
                    if s_id == key: gold_facts.append(s_text.strip())
        except: continue
    return " ".join(gold_facts) if gold_facts else ex.get("response", "")

def load_techqa_samples(path, size, seed):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    random.seed(seed)
    return [json.loads(l) for l in random.sample(lines, min(size, len(lines)))]

# =================================================
# 3. Main Experiment Loop
# =================================================
async def main():
    data_samples = load_techqa_samples(DATA_PATH, SAMPLE_SIZE, SEED)
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # 중복 체크 세트 로드
    processed_keys = get_processed_keys(OUTPUT_JSONL)
    print(f"[INFO] Processed: {len(processed_keys)} samples. Resuming...")

    # 파일 모드를 'a' (Append)로 열기
    csv_mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
    csv_f = open(OUTPUT_CSV, csv_mode, newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_f)
    if csv_mode == "w":
        writer.writerow(["model", "retrieval", "chunk", "top_k", "status"])

    jsonl_f = open(OUTPUT_JSONL, "a", encoding="utf-8")

    for MODEL_NAME in MODELS:
        engine_rag = InferenceEngine("NAIVE_RAG", MODEL_NAME, verbose=False)

        for chunk_name, paths in CHUNK_CONFIGS.items():
            if not os.path.exists(paths["bm25"]): continue # 경로 안정성 강화
            
            with open(paths["bm25"], "rb") as f:
                bm_data = pickle.load(f)
                bm25 = bm_data["bm25"]
                bm25_corpus = [m["text"] for m in bm_data["metas"]]

            faiss_index = faiss.read_index(paths["faiss"])
            with open(paths["meta"], "rb") as f:
                vec_corpus = [m["text"] for m in pickle.load(f)]

            for top_k in TOP_K_LIST:
                for retrieval_mode in ["BM25", "VEC"]:
                    desc = f"{MODEL_NAME[:8]} | {retrieval_mode} | {chunk_name} | K={top_k}"
                    
                    for ex in tqdm(data_samples, desc=desc):
                        q_id = ex["id"]
                        
                        # 중복 검사: 이미 처리했다면 스킵
                        current_key = (MODEL_NAME, retrieval_mode, chunk_name, top_k, q_id)
                        if current_key in processed_keys:
                            continue

                        q_text = ex["question"]
                        ground_truth = extract_gold_standard_b(ex)

                        # 1. Retrieval
                        if retrieval_mode == "BM25":
                            scores = bm25.get_scores(q_text.split())
                            idxs = np.argsort(scores)[::-1][:top_k]
                            chunks = [bm25_corpus[i] for i in idxs]
                        else:
                            q_emb = embedder.encode([q_text], normalize_embeddings=True).astype("float32")
                            _, idxs = faiss_index.search(q_emb, top_k)
                            chunks = [vec_corpus[i] for i in idxs[0] if i != -1]

                        # 2. 추론
                        prediction = await engine_rag(q_text, chunks)

                        # 3. 저장
                        log_entry = {
                            "metadata": {
                                "id": q_id, "model": MODEL_NAME, "retrieval": retrieval_mode,
                                "chunk_config": chunk_name, "top_k": top_k
                            },
                            "question": q_text,
                            "contexts": chunks,
                            "prediction": prediction,
                            "ground_truth": ground_truth,
                            "original_response": ex.get("response", "")
                        }
                        jsonl_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        jsonl_f.flush()

                    writer.writerow([MODEL_NAME, retrieval_mode, chunk_name, top_k, "COMPLETED"])
                    csv_f.flush()

    csv_f.close()
    jsonl_f.close()
    print(f"\n[DONE] Experiment resumed and completed. Results: {OUTPUT_JSONL}")

if __name__ == "__main__":
    asyncio.run(main())