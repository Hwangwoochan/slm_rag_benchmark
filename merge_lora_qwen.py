import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 경로 설정 (사용자님의 환경에 맞게 수정하세요)
base_path = os.path.expanduser("~/Desktop/llama.cpp/Qwen2.5-0.5B-Instruct")
adapter_path = "outputs/qwen2.5_0.5b_rag_lora"
save_path = "final_merged_qwen_rag"

print("1. Qwen 베이스 모델 로드 중 (FP16)...")
# Qwen은 trust_remote_code=True를 넣어주는 것이 안전합니다.
base_model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    torch_dtype=torch.float16, 
    device_map="cpu",
    trust_remote_code=True
)

print("2. Qwen 어댑터 적용 및 병합 중...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# [중요] 용량 뻥튀기 방지를 위해 다시 한번 FP16 강제 고정
merged_model = merged_model.half() 

print(f"3. 최종 병합 모델 저장 중 -> {save_path}")
merged_model.save_pretrained(save_path, safe_serialization=True)

# 토크나이저도 함께 저장
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print(f"✅ Qwen 병합 완료! 폴더를 확인하세요: {save_path}")