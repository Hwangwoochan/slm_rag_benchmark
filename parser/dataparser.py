from datasets import load_dataset
import json

# =====================
# 1. PopQA entity 추출
# =====================
popqa = load_dataset("akariasai/PopQA", split="test")

entity_titles = set()
for ex in popqa:
    title = ex.get("s_wiki_title")
    if title:
        entity_titles.add(title)

print(f"Loaded {len(entity_titles)} unique entity titles")

# =====================
# 2. Wikipedia streaming
# =====================
wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

output_path = "popqa_entity_wiki.jsonl"
doc_counter = 0

with open(output_path, "w", encoding="utf-8") as f:
    for doc in wiki:
        title = doc["title"]

        if title not in entity_titles:
            continue

        text = doc["text"].strip()
        if len(text) < 300:
            continue

        doc_id = f"doc_{doc_counter:07d}"

        f.write(json.dumps({
            "doc_id": doc_id,    
            "title": title,
            "entity": title,       # PopQA entity
            "text": text,
            "source": "wikipedia"
        }, ensure_ascii=False) + "\n")

        doc_counter += 1

print(f"Saved {doc_counter} entity pages")
