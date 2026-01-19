import json
import os
from tqdm import tqdm
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

# 1. 경로 및 설정
# INPUT_PATH = "./rag_techqa_llm_judge_data_W.jsonl"
# OUTPUT_PATH = "./rag_techqa_llm_judge_data_W_eval_results.jsonl"


INPUT_PATH = "./rag_techqa_llm_judge_data_leaf.jsonl"
OUTPUT_PATH = "./leaf_eval_results.jsonl"

# 2. Prometheus 모델 로드 (vLLM 최적화 설정 추가)
print("Initializing Prometheus Model on GPU...")
model = VLLM(
    model="prometheus-eval/prometheus-7b-v2.0",
)
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

# 3. 영문 루브릭 정의 (그대로 유지)
rubric_data = {
    "criteria": "How accurately and completely does the SLM's response include the core information and technical details specified in the Ground Truth (Reference Answer)?",
    "score1_description": "The response is entirely unhelpful, refuses to answer, or provides 'Hallucinated' information that is irrelevant to or contradicts the Ground Truth document.",
    "score2_description": "The response identifies the user's intent but fails to include most key solutions, specific configuration values, or essential technical details found in the Ground Truth.",
    "score3_description": "The response mentions the primary solution but is partially incomplete. Specific steps, property names, or technical parameters are inaccurate or missing.",
    "score4_description": "The response includes almost all content from the Ground Truth and is technically accurate, with only minor omissions or slight differences in terminology phrasing.",
    "score5_description": "The response perfectly incorporates all key elements (configuration values, procedures, definitions, etc.) from the Ground Truth, demonstrating high technical proficiency and clarity."
}
score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

# 4. 파일 읽기
print(f"Reading data from {INPUT_PATH}...")
samples = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

# 5, 6, 7. 개별 평가 수행 및 실시간 저장
print(f"Starting sequential evaluation for {len(samples)} samples...")

# 결과를 저장할 디렉토리가 없다면 생성
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    # tqdm으로 진행 상태를 실시간으로 확인 (87% 멈춤 현상 파악 가능)
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            # single_absolute_grade를 사용하여 파라미터 에러 원천 차단
            feedback, score = judge.single_absolute_grade(
                instruction=sample['question'],
                response=sample['prediction'],
                rubric=score_rubric,
                reference_answer=sample['ground_truth']
            )
            
            # 결과 데이터 업데이트
            sample['prometheus_score'] = score
            sample['prometheus_feedback'] = feedback
            
            # 한 줄씩 즉시 저장 (프로그램이 죽어도 데이터 보존)
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush() # 버퍼 강제 비우기
            
        except Exception as e:
            print(f"\nError processing ID {sample.get('metadata', {}).get('id')}: {e}")
            continue

print(f"\nEvaluation successfully completed! Results saved to {OUTPUT_PATH}")