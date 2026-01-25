import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------------
# 1. 환경 설정 및 하이퍼파라미터
# -----------------------------
MAX_LEN = 1024
BASE_MODEL_DIR = os.path.expanduser("~/Desktop/llama.cpp/smollm2_135m_instruct")
TRAIN_JSONL = "data/rola_data/train_oracle_plus_idk_ver2.jsonl"
OUT_DIR = "outputs/smollm2_135m_rag_lora"

# -----------------------------
# 2. 템플릿 일치화 함수 (NAIVE_RAG_PROMPT와 동일 구조)
# -----------------------------
def build_text(ex):
    p = ex["prompt"].rstrip()
    r = ex["response"].rstrip()
    c = (ex.get("context") or "").rstrip()

    # 추론 프롬프트와 완벽히 일치하는 구조로 생성
    if c:
        text = (
            "You are a careful assistant.\n\n"
            "Rules:\n"
            "- Use ONLY the provided context.\n"
            "- Do NOT use external knowledge or assumptions.\n"
            "- Answer concisely (1–3 sentences).\n"
            "- If the answer cannot be determined from the context, say exactly: \"I don't know\".\n\n"
            "Context:\n"
            f"{c}\n\n"
            "Question:\n"
            f"{p}\n\n"
            "Output format:\n"
            "Answer: <1–3 sentences>\n"
            "Evidence: [<sentence_ids>]\n\n"
            "Answer:\n" # 추론 시 입력이 끝나는 지점
            f"{r}"       # 모델이 생성해야 할 정답
        )
    else:
        # Context가 없는 경우 (ONLY_SLM 대응)
        text = (
            "Answer the following question.\n\n"
            "Rules:\n"
            "- Answer concisely (1–3 sentences).\n"
            "- Be factual.\n"
            "- If you do not know the answer, say exactly: \"I don't know\".\n\n"
            f"Question:\n{p}\n\n"
            "Answer:\n"
            f"{r}"
        )
    return {"text": text}

def main():
    # 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 모델 로드 (dtype 인자 사용)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    # 데이터 로드 및 텍스트 변환
    ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    ds = ds.map(build_text)

    # [중요] 미리 토큰화하여 SFTConfig 인자 에러 원천 차단
    def tokenize_fn(batch):
        out = tok(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    # 토큰화 후 불필요한 원본 컬럼 제거
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # LoRA 설정 (모든 선형 레이어 학습)
    lora_cfg = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
    )

    # 학습 설정 (순수 학습 관련 인자만 유지)
    sft_cfg = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
        eval_strategy="no",
    )

    # 트레이너 실행
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=sft_cfg,
        processing_class=tok,
    )

    print("--- 🚀 학습 시작 (템플릿 일치 완료) 🚀 ---")
    trainer.train()
    
    # 저장
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"[OK] 모델과 어댑터가 저장되었습니다: {OUT_DIR}")

if __name__ == "__main__":
    main()