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
DOC_PATH = "./data/popqa_entity_wiki.jsonl"
OUT_BASE = Path("./data/mdbr_teacher_vector_indices")
OUT_BASE.mkdir(exist_ok=True)

CHUNK_CONFIGS = {
    # "W100": {
    #     "type": "word",
    #     "size": 100,
    #     "overlap": 0
    # },
    # "T256_O64": {
    #     "type": "token",
    #     "size": 256,
    #     "overlap": 64
    # },
    "T512_O128": {
        "type": "token",
        "size": 512,
        "overlap": 128
    }
}

enc = tiktoken.get_encoding("cl100k_base")

#model = SentenceTransformer("all-MiniLM-L6-v2")

# model = SentenceTransformer("MongoDB/mdbr-leaf-ir")

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
        chunks.append(enc.decode(tokens[i:i+size]))
        i += size - overlap
    return chunks

# =====================
# 문서 로드
# =====================
with open(DOC_PATH, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

print(f"[INFO] Loaded {len(docs)} documents")

# =====================
# Chunking × Vector Index
# =====================
for mode, cfg in CHUNK_CONFIGS.items():
    print(f"\n=== Building Vector Index for {mode} ===")

    metas = []
    texts = []

    for doc in docs:
        doc_id = doc["doc_id"]
        entity = doc.get("entity", doc_id)
        text = doc["text"]

        if cfg["type"] == "word":
            chunks = chunk_words(text, cfg["size"])
        else:
            chunks = chunk_tokens(text, cfg["size"], cfg["overlap"])

        for idx, ch in enumerate(chunks):
            metas.append({
                "chunk_id": f"{doc_id}::{mode}::{idx}",
                "doc_id": doc_id,
                "entity": entity,
                "text": ch,
                "chunk_meta": {
                    "chunk_type": cfg["type"],
                    "chunk_size": cfg["size"],
                    "chunk_overlap": cfg["overlap"],
                    "chunk_index": idx
                }
            })
            texts.append(ch)

    print(f"[INFO] Total chunks: {len(texts)}")

    # =====================
    # Embedding
    # =====================
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embs = np.array(embs).astype("float32")

    # =====================
    # FAISS index
    # =====================
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # =====================
    # 저장
    # =====================
    out_dir = OUT_BASE / mode
    out_dir.mkdir(exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)

    print(f"[DONE] Saved → {out_dir}")
