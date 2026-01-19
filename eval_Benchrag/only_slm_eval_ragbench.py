import json
import asyncio
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset # 또는 로컬 파일 읽기

from utility.inference import InferenceEngine

# =================================================
# 1. 설정 (Configuration)
# =====================
MODELS = [
    "smollm2:135m",
    "qwen2.5:0.5b",
    "qwen2.5:7b",
]

DATA_PATH = "./data/combined_data.jsonl"  # 데이터 경로
OUTPUT_DIR = Path("./results/techqa_inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 1000  # 추론할 질문 개수
SEED = 42

# =================================================
# 2. TechQA 파서 (Parser)
# =================================================
def techqa_parser(line):
    """JSONL 한 줄을 읽어 ID와 질문을 반환"""
    data = json.loads(line)
    return {
        "id": data.get("id"),
        "question": data.get("question"),
        "source": data.get("origin_dataset", "techqa")
    }

# =================================================
# 3. 메인 추론 루프
# =================================================
async def main():
    # 데이터 로드 (JSONL 파일 읽기)
    print(f"[INFO] Reading data from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 샘플링 (고정된 SEED 사용)
    import random
    random.seed(SEED)
    if len(lines) > MAX_SAMPLES:
        lines = random.sample(lines, MAX_SAMPLES)

    for model_name in MODELS:
        print(f"\n>>>> Starting Inference: {model_name}")
        # 파일명에서 특수문자 제거 (: -> _)
        safe_model_name = model_name.replace(":", "_")
        output_file = OUTPUT_DIR / f"{safe_model_name}_results.jsonl"
        
        # 엔진 초기화 (RAG 없이 모델 지식만 사용)
        engine = InferenceEngine(mode="ONLY_SLM", model=model_name, verbose=False)
        
        # 결과 저장 파일 오픈
        with open(output_file, "w", encoding="utf-8") as out_f:
            for line in tqdm(lines, desc=f"Processing {model_name}"):
                # 1. 파싱
                item = techqa_parser(line)
                q_id = item["id"]
                question = item["question"]

                # 2. 모델 추론
                try:
                    prediction = await engine(question=question)
                except Exception as e:
                    print(f"Error at {q_id}: {e}")
                    prediction = "ERROR_INFERENCE"

                # 3. 결과 저장 (JSONL 형식)
                result_entry = {
                    "id": q_id,
                    "model": model_name,
                    "question": question,
                    "prediction": prediction
                }
                out_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                out_f.flush() # 실시간 기록

    print(f"\n[DONE] All inferences completed. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())