import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =====================
# 설정
# =====================
DOC_PATH = "./data/popqa_entity_wiki.jsonl"
OUT_BASE = Path("./data/mdbr_teacher_vector_indices_smallsize")
OUT_BASE.mkdir(exist_ok=True, parents=True)

# 모델 로드 (v1.5는 대형 모델이므로 가능하면 GPU 사용)
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5", device="cuda")

# 512 토큰 제한을 고려한 설정
CHUNK_CONFIGS = {
    "T512_O128": {
        "size": 128,
        "overlap": 32
    },
    "W100": {
        "type": "word",
        "size": 50,
        "overlap": 0
    },
}

# =====================
# Chunking 함수 (모델 전용 토크나이저 활용)
# =====================
def chunk_tokens_native(text, size, overlap):
    """모델 고유 토크나이저를 사용하여 문맥 손실 최소화"""
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + size]
        chunks.append(model.tokenizer.decode(chunk_tokens))
        if i + size >= len(tokens): break
        i += size - overlap
    return chunks

# =====================
# 메인 프로세스
# =====================
with open(DOC_PATH, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]
print(f"[INFO] Loaded {len(docs)} documents")

for mode, cfg in CHUNK_CONFIGS.items():
    print(f"\n=== Building Vector Index for {mode} ===")

    metas = []
    texts = []

    # 1. Chunking
    for doc in tqdm(docs, desc="Chunking"):
        doc_id = doc["doc_id"]
        entity = doc.get("entity", doc_id)
        
        # Arctic 모델의 토크나이저를 직접 사용하여 더 정확하게 분할
        chunks = chunk_tokens_native(doc["text"], cfg["size"], cfg["overlap"])

        for idx, ch in enumerate(chunks):
            metas.append({
                "chunk_id": f"{doc_id}::{mode}::{idx}",
                "doc_id": doc_id,
                "entity": entity,
                "text": ch,
                "chunk_meta": { "chunk_index": idx, **cfg }
            })
            texts.append(ch)

    print(f"[INFO] Total chunks: {len(texts)}")

    # 2. Embedding (Snowflake Arctic 최적화)
    # 문서를 인덱싱할 때는 Prefix를 붙이지 않습니다.
    embs = model.encode(
        texts,
        batch_size=64, # 환경에 따라 조정
        normalize_embeddings=True, # IP 인덱스 사용을 위해 필수
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embs = embs.astype("float32")

    # 3. FAISS Index 구축
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim) # 코사인 유사도를 위해 Inner Product 사용
    index.add(embs)

    # 4. 저장
    out_dir = OUT_BASE / mode
    out_dir.mkdir(exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)

    print(f"[DONE] Saved → {out_dir}")