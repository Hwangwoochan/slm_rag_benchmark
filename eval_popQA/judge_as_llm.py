import google.generativeai as genai
import json
import time
import os
from tqdm import tqdm

# =================================================
# 1. 설정 (Configuration)
# =================================================
# 발급받은 API 키를 여기에 입력하세요.
API_KEY = "AIzaSyBXh8qkSH0mb-_EJWBu-3-kpHEwzh5fwHQ" 
genai.configure(api_key=API_KEY)

# 추론 능력이 뛰어난 gemini-1.5-pro 모델 사용 권장 [cite: 450, 481]
model = genai.GenerativeModel(
    'gemini-1.5-pro',
    generation_config={"response_mime_type": "application/json"}
)

INPUT_FILE = "rag_techqa_llm_judge_data_method_b_new.jsonl"
OUTPUT_FILE = "final_evaluated_results.jsonl"

# =================================================
# 2. 고도화된 평가 프롬프트 (Chain-of-Thought 반영)
# =================================================
def get_eval_prompt(item):
    return f"""
너는 최고의 기술 지원 품질 평가관이야. 제공된 [Gold Evidence]를 바탕으로 모델의 [Prediction]을 엄격히 평가해줘.

[데이터]
- 질문: {item['question']}
- Gold Evidence (절대적 팩트): {item['ground_truth']}
- 모델의 Prediction (평가 대상): {item['prediction']}

[평가 단계]
1. 분석(Analysis): 모델의 답변이 Gold Evidence의 핵심 해결책(제품명, 기술 단계, 주의사항 등)을 얼마나 정확하게 포함하는지 분석하라. [cite: 470, 489]
2. 정확성(Accuracy): 분석을 바탕으로 1~5점 점수를 매겨라. (정답과 반대되는 대답은 1점) [cite: 457]
3. 근거성(Groundedness): 모델이 외부 지식을 끌어오거나 허구의 내용을 지어내지는 않았는지 확인하여 1~5점 점수를 매겨라. [cite: 462]

[Accuracy 점수 기준]
- 5점: 모든 기술적 해결책이 팩트와 일치하며 완벽함.
- 3점: 핵심 방향은 맞으나 구체적인 단계나 연락처 등이 누락됨.
- 1점: 정답과 정반대로 말하거나 완전한 할루시네이션(환각).

반드시 아래 JSON 형식으로만 응답하라:
{{
  "thought": "여기에 상세한 논리적 분석 과정을 적으시오",
  "accuracy": 점수,
  "groundedness": 점수,
  "reason": "최종 점수에 대한 요약 비평"
}}
"""

# =================================================
# 3. 평가 실행 (Evaluation Loop)
# =================================================
def run_evaluation():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] 입력 파일이 없습니다: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[INFO] {len(lines)}개의 샘플에 대해 LLM-as-a-Judge 평가를 시작합니다.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines):
            item = json.loads(line)
            prompt = get_eval_prompt(item)
            
            try:
                # Gemini API 호출
                response = model.generate_content(prompt)
                
                # JSON 응답 파싱
                # Gemini가 간혹 마크다운 태그를 포함할 수 있으므로 정제 후 로드
                clean_response = response.text.strip().replace('```json', '').replace('```', '')
                eval_res = json.loads(clean_response)
                
                # 결과 통합 저장
                item['llm_thought'] = eval_res.get('thought', "")
                item['score_accuracy'] = eval_res.get('accuracy', 0)
                item['score_groundedness'] = eval_res.get('groundedness', 0)
                item['eval_reason'] = eval_res.get('reason', "")
                
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
                
                # 무료 티어 RPM 제한 준수 (약 4~5초 대기)
                time.sleep(4) 
                
            except Exception as e:
                print(f"\n[ERROR] ID {item['metadata']['id']} 처리 중 오류 발생: {e}")
                time.sleep(10) # 오류 발생 시 더 오래 대기
                continue

    print(f"\n[SUCCESS] 평가가 완료되었습니다! 결과 파일: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()