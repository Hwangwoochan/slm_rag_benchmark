import json
import time
import os
from google import genai
from google.genai import types
from tqdm import tqdm

# =================================================
# 1. 설정 (사용자 목록에서 확인된 최신 모델 적용)
# =================================================
API_KEY = "AIzaSyBXh8qkSH0mb-_EJWBu-3-kpHEwzh5fwHQ" 
client = genai.Client(api_key=API_KEY)

# 목록에서 확인된 최신 Pro 모델 ID 사용
EVAL_MODEL = 'gemini-2.5-pro' 

INPUT_FILE = "./results/techqa_inference/smollm2_135m_results.jsonl"
OUTPUT_FILE = "./results/techqa_inference/final_evaluated_smollm.jsonl"

# =================================================
# 2. 채점 로직 (CoT 반영)
# =================================================
def get_eval_prompt(item):
    return f"""
너는 세계 최고의 기술 지원 평가 전문가야. [Gold Truth]를 기준으로 [Prediction]을 엄격하게 심사하라.

[데이터]
- 질문: {item['question']}
- 정답 팩트(GT): {item['ground_truth']}
- 모델 답변(Pred): {item['prediction']}

[심사 단계]
1. Thought: 모델 답변이 GT의 핵심 기술 단계를 모두 포함하는지, 환각(지어낸 말)은 없는지 분석하라.
2. Accuracy: 정답과 일치 정도 (1~5점)
3. Faithfulness: 주어진 정보에만 충실했는지 여부 (1~5점)

반드시 JSON으로만 응답하라:
{{
  "thought": "분석 내용",
  "accuracy": 점수,
  "faithfulness": 점수,
  "reason": "최종 비평"
}}
"""

async def run_evaluation():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[INFO] {EVAL_MODEL} 모델로 채점을 시작합니다. (총 {len(lines)}개)")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines):
            item = json.loads(line)
            
            try:
                response = client.models.generate_content(
                    model=EVAL_MODEL,
                    contents=get_eval_prompt(item),
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json'
                    )
                )
                
                eval_res = json.loads(response.text)
                item.update({
                    'eval_accuracy': eval_res.get('accuracy'),
                    'eval_faithfulness': eval_res.get('faithfulness'),
                    'eval_thought': eval_res.get('thought'),
                    'eval_reason': eval_res.get('reason')
                })
                
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(4) # 무료 티어 속도 제한(RPM) 준수

            except Exception as e:
                print(f"\nID {item.get('id')} 오류: {e}")
                continue

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluation())