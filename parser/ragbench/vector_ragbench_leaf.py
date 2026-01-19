import json
import pickle
from pathlib import Path
import numpy as np
import faiss
import tiktoken
from sentence_transformers import SentenceTransformer

# =====================
# 설정
# =====================
# 통합된 데이터셋 경로
DOC_PATH = "./data/combined_data.jsonl" 
OUT_BASE = Path("./data/leaf_minilm_vector_indices_combined")
OUT_BASE.mkdir(exist_ok=True, parents=True)

# 3가지 청킹 설정 유지
CHUNK_CONFIGS = {
   "W100": {
        "type": "word",
        "size": 100,
        "overlap": 0
    }
}

enc = tiktoken.get_encoding("cl100k_base")

 # 요청하신 모델로 변경
print("[INFO] Loading embedding model: Snowflake/snowflake-arctic-embed-m-v1.5")
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")

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
        chunk_tokens_list = tokens[i : i + size]
        chunks.append(enc.decode(chunk_tokens_list))
        i += (size - overlap)
        if i >= len(tokens): break
    return chunks

# =====================
# 데이터 로드 (JSONL 통합 데이터)
# =====================
data = []
print(f"[INFO] Reading integrated file: {DOC_PATH}")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"[INFO] Loaded {len(data)} integrated samples")

# =====================
# Chunking × Vector Index 빌드 루프
# =====================
for mode, cfg in CHUNK_CONFIGS.items():
    print(f"\n=== Building Vector Index for {mode} ===")

    metas = []
    texts = []

    for entry in data:
        entry_id = entry["id"]
        # 통합 시 추가한 출처 정보 (없으면 unknown)
        source = entry.get("origin_dataset", "unknown")
        
        # 중첩된 'documents' 리스트 순회 처리
        for doc_idx, text in enumerate(entry.get("documents", [])):
            if not text or not text.strip():
                continue

            # 설정에 따른 청킹
            if cfg["type"] == "word":
                chunks = chunk_words(text, cfg["size"])
            else:
                chunks = chunk_tokens(text, cfg["size"], cfg["overlap"])

            for idx, ch in enumerate(chunks):
                metas.append({
                    "chunk_id": f"{source}::{entry_id}::doc{doc_idx}::{idx}",
                    "entry_id": entry_id,
                    "doc_index": doc_idx,
                    "source": source,
                    "text": ch,
                    "chunk_meta": {
                        "chunk_type": cfg["type"],
                        "chunk_size": cfg["size"],
                        "chunk_index": idx
                    }
                })
                texts.append(ch)

    print(f"[INFO] Total chunks created for {mode}: {len(texts)}")

    # =====================
    # Embedding
    # =====================
    print(f"[INFO] Embedding {len(texts)} chunks...")
    embs = model.encode(
        texts,
        batch_size=64, # MiniLM은 가벼워서 배치를 더 크게 잡으셔도 됩니다.
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embs = np.array(embs).astype("float32")

    # =====================
    # FAISS index
    # =====================
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim) # 코사인 유사도 검색
    index.add(embs)

    # =====================
    # 저장
    # =====================
    out_dir = OUT_BASE / mode
    out_dir.mkdir(exist_ok=True, parents=True)

    faiss.write_index(index, str(out_dir / "faiss_combined.index"))
    with open(out_dir / "metas_combined.pkl", "wb") as f:
        pickle.dump(metas, f)

    print(f"[DONE] Combined Index for {mode} saved → {out_dir}")