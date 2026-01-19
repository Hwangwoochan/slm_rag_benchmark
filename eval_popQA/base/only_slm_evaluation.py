import re
import json
import asyncio
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from utility.inference import InferenceEngine

# =================================================
# 1. Config
# =================================================
MODELS = [
    "smollm2:135m",
    "qwen2.5:0.5b",
    "qwen2.5:7b",
]

SAMPLE_SIZE = 100  # 랜덤으로 선택할 질문 개수
SEED = 42
OUTPUT_CSV = "only_slm_eval_results.csv"

# =================================================
# 2. Utils & Metrics
# =================================================
def normalize(s: str) -> str:
    """텍스트 정규화: 소문자화 및 특수문자 제거"""
    return re.sub(r"\W+", " ", str(s).lower()).strip()

def parse_answers(ans) -> list:
    """PopQA 정답 리스트 파싱"""
    if ans is None: return []
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except:
            return [normalize(ans)]
    
    flat = []
    if isinstance(ans, list):
        for a in ans:
            if isinstance(a, list): flat.extend(a)
            else: flat.append(a)
    else:
        flat.append(ans)
    return list({normalize(a) for a in flat if normalize(a)})

def compute_metrics(pred: str, answers: list[str]):
    """Accuracy(EM 기반) 및 Token F1 계산"""
    pred_norm = normalize(pred)
    pred_toks = set(pred_norm.split())
    
    acc = 0.0
    best_f1 = 0.0
    
    for a in answers:
        a_norm = normalize(a)
        # 1. Accuracy (정답이 포함되어 있는가)
        if a_norm in pred_norm:
            acc = 1.0
            
        # 2. Token F1
        ans_toks = set(a_norm.split())
        if not ans_toks or not pred_toks:
            continue
            
        common = pred_toks & ans_toks
        if not common:
            continue
            
        p = len(common) / len(pred_toks)
        r = len(common) / len(ans_toks)
        f1 = 2 * p * r / (p + r)
        best_f1 = max(best_f1, f1)
        
    # 만약 Accuracy가 1이면 F1도 1로 간주 (필요에 따라 조정 가능)
    if acc == 1.0:
        best_f1 = max(best_f1, 1.0)
        
    return acc, best_f1

# =================================================
# 3. Main Evaluation
# =================================================
async def main():
    # 데이터셋 로드 및 랜덤 샘플링
    print(f"[INFO] Loading PopQA and sampling {SAMPLE_SIZE} questions...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    popqa = popqa.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    final_report = []

    for model_name in MODELS:
        print(f"\n>>>> Evaluating Model: {model_name}")
        # ONLY_SLM 모드로 엔진 초기화 (RAG 사용 안 함)
        engine = InferenceEngine(mode="ONLY_SLM", model=model_name, verbose=False)
        
        acc_list = []
        f1_list = []

        for ex in tqdm(popqa, desc=f"Testing {model_name}"):
            q = ex["question"]
            answers = parse_answers(ex.get("possible_answers", []))
            if not answers:
                continue

            # 모델 추론
            prediction = await engine(question=q)
            
            # 지표 계산
            acc, f1 = compute_metrics(prediction, answers)
            acc_list.append(acc)
            f1_list.append(f1)

        avg_acc = np.mean(acc_list)
        avg_f1 = np.mean(f1_list)
        
        print(f"[{model_name}] Avg Accuracy: {avg_acc:.4f} | Avg F1: {avg_f1:.4f}")
        
        final_report.append({
            "model": model_name,
            "avg_accuracy": avg_acc,
            "avg_f1": avg_f1,
            "samples": len(acc_list)
        })

    # 결과 저장 및 출력
    df = pd.DataFrame(final_report)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(df)
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())