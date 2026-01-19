import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import tiktoken

# =====================
# 설정
# =====================
# RAGBench/TechQA 데이터는 보통 줄바꿈으로 구분된 JSONL 형식입니다.
DOC_PATH = "./data/combined_data.jsonl" 
OUT_BASE = Path("./data/bm25_indices_techqa")
OUT_BASE.mkdir(exist_ok=True)

CHUNK_CONFIGS = {
    "W100": {
        "type": "word",
        "size": 100,
        "overlap": 0
    },
    "T256_O64": {
        "type": "token",
        "size": 256,
        "overlap": 64
    },
    "T512_O128": {
        "type": "token",
        "size": 512,
        "overlap": 128
    }
}

enc = tiktoken.get_encoding("cl100k_base")

# =====================
# Chunking 함수
# =====================
def chunk_words(text, size):
    words = text.split()
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
    ]

def chunk_tokens(text, size, overlap):
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        # 토큰을 잘라내고 다시 텍스트로 디코딩
        chunk_tokens_list = tokens[i : i + size]
        chunks.append(enc.decode(chunk_tokens_list))
        i += (size - overlap)
        if i >= len(tokens): break
    return chunks

# =====================
# 데이터 로드 (JSONL 대응)
# =====================
# json.load(f) 대신 한 줄씩 json.loads()를 수행하여 "Extra data" 에러 해결
data = []
print(f"[INFO] Reading file: {DOC_PATH}")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"[INFO] Loaded {len(data)} question samples")

# =====================
# Chunking × BM25 빌드
# =====================
for mode, cfg in CHUNK_CONFIGS.items():
    print(f"\n=== Building BM25 for {mode} ===")

    corpus_tokens = []
    metas = []
    chunk_counter = 0

    for entry in data:
        entry_id = entry["id"]      # 예: "techqa_TRAIN_Q418"
        question = entry["question"]
        
        # TechQA의 핵심: documents 리스트 순회
        # 한 질문에 딸린 여러 개의 문서를 각각 청킹합니다.
        for doc_idx, text in enumerate(entry.get("documents", [])):
            if not text or not text.strip():
                continue

            # 설정된 방식에 따른 청킹 수행
            if cfg["type"] == "word":
                chunks = chunk_words(text, cfg["size"])
            else:
                chunks = chunk_tokens(text, cfg["size"], cfg["overlap"])

            for idx, ch in enumerate(chunks):
                # BM25는 단어 리스트(토큰화된 상태)를 입력으로 받습니다.
                corpus_tokens.append(ch.split())
                metas.append({
                    "chunk_id": f"{entry_id}::doc{doc_idx}::{mode}::{idx}",
                    "entry_id": entry_id,
                    "doc_index": doc_idx,
                    "text": ch,
                    "chunk_meta": {
                        "chunk_type": cfg["type"],
                        "chunk_size": cfg["size"],
                        "chunk_overlap": cfg["overlap"],
                        "chunk_index": idx
                    }
                })
                chunk_counter += 1

    print(f"[INFO] Total chunks for {mode}: {chunk_counter}")

    # BM25 인덱스 생성
    bm25 = BM25Okapi(corpus_tokens)

    # 결과 저장
    out_dir = OUT_BASE / mode
    out_dir.mkdir(exist_ok=True)
    
    save_path = out_dir / "bm25.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "metas": metas,
            "chunk_config": cfg
        }, f)

    print(f"[DONE] Saved → {save_path}")