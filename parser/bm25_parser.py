import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import tiktoken

# =====================
# 설정
# =====================
DOC_PATH = "popqa_entity_wiki.jsonl"
OUT_BASE = Path("bm25_indices")
OUT_BASE.mkdir(exist_ok=True)

# 3가지 chunking 세팅
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
# Chunking × BM25
# =====================
for mode, cfg in CHUNK_CONFIGS.items():
    print(f"\n=== Building BM25 for {mode} ===")

    corpus_tokens = []
    metas = []
    chunk_counter = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        entity = doc.get("entity", doc_id)
        text = doc["text"]

        if cfg["type"] == "word":
            chunks = chunk_words(text, cfg["size"])
        else:
            chunks = chunk_tokens(text, cfg["size"], cfg["overlap"])

        for idx, ch in enumerate(chunks):
            corpus_tokens.append(ch.split())
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
            chunk_counter += 1

    print(f"[INFO] Total chunks: {chunk_counter}")

    bm25 = BM25Okapi(corpus_tokens)

    out_dir = OUT_BASE / mode
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "metas": metas,
            "chunk_config": cfg
        }, f)

    print(f"[DONE] Saved → {out_dir / 'bm25.pkl'}")
