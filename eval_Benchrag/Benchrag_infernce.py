import json
import asyncio
import re
import random
from pathlib import Path
from tqdm import tqdm

from utility.inference import InferenceEngine

# =================================================
# 1. 설정 (Configuration)
# =================================================
MODELS = ["smollm2:135m", "qwen2.5:0.5b", "qwen2.5:7b"]
DATA_PATH = "./data/combined_data.jsonl"
OUTPUT_DIR = Path("./results/techqa_inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 100 # 평가를 위해 100개 권장
SEED = 42

# =================================================
# 2. 방법 B: 팩트 기반 정답 추출 로직 [cite: 207, 212]
# =================================================
def extract_gold_standard_b(data):
    """원시 데이터에서 팩트 문장들을 추출하여 GT 생성"""
    relevant_keys = data.get("all_relevant_sentence_keys", [])
    doc_sentences = data.get("documents_sentences", [])
    
    if not relevant_keys or not doc_sentences:
        return data.get("response", "No gold standard available.")

    gold_facts = []
    for key in relevant_keys:
        try:
            doc_idx_match = re.search(r'\d+', key)
            if not doc_idx_match: continue
            doc_idx = int(doc_idx_match.group())
            
            if doc_idx < len(doc_sentences):
                for s_id, s_text in doc_sentences[doc_idx]:
                    if s_id == key:
                        gold_facts.append(s_text.strip())
        except (IndexError, ValueError, AttributeError):
            continue
            
    return " ".join(gold_facts) if gold_facts else data.get("response", "")

# =================================================
# 3. 메인 추론 루프
# =================================================
async def main():
    print(f"[INFO] Reading data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    random.seed(SEED)
    if len(lines) > MAX_SAMPLES:
        lines = random.sample(lines, MAX_SAMPLES)

    for model_name in MODELS:
        print(f"\n>>>> Starting Inference: {model_name}")
        safe_model_name = model_name.replace(":", "_")
        output_file = OUTPUT_DIR / f"{safe_model_name}_results.jsonl"
        
        # 엔진 초기화
        engine = InferenceEngine(mode="ONLY_SLM", model=model_name, verbose=False)
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            for line in tqdm(lines, desc=f"Processing {model_name}"):
                data = json.loads(line)
                q_id = data.get("id")
                question = data.get("question")
                
                # 1. 정답(GT) 미리 추출 (방법 B 적용)
                ground_truth = extract_gold_standard_b(data)
                original_response = data.get("response", "")

                # 2. 모델 추론
                try:
                    prediction = await engine(question=question)
                except Exception as e:
                    print(f"Error at {q_id}: {e}")
                    prediction = "ERROR_INFERENCE"

                # 3. 결과 저장 (평가에 필요한 모든 정보 포함)
                result_entry = {
                    "metadata": {
                        "id": q_id,
                        "model": model_name,
                        "source": data.get("origin_dataset", "techqa")
                    },
                    "question": question,
                    "prediction": prediction,
                    "ground_truth": ground_truth,      # Gemini가 비교할 실제 팩트
                    "original_response": original_response # 참고용 모범 답안
                }
                out_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\n[DONE] Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())