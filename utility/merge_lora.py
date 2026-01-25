from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

base_path = os.path.expanduser("~/Desktop/llama.cpp/smollm2_135m_instruct")
adapter_path = "outputs/smollm2_135m_rag_lora"
save_path = "final_merged_smollm2_rag"

print("1. 베이스 모델 로드 중 (FP16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    torch_dtype=torch.float16, 
    device_map="cpu"
)

print("2. 어댑터 적용 및 병합 중...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# [핵심 수정] 저장 전 다시 한번 FP16으로 강제 변환
merged_model = merged_model.half() 

print(f"3. 최종 모델 저장 중 -> {save_path}")
# safe_serialization=True를 권장합니다.
merged_model.save_pretrained(save_path, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(save_path)

print("✅ 병합 완료! 이제 다시 GGUF 변환을 시도해보세요.")