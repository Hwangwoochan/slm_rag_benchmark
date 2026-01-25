import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------------
# 1. 환경 설정 (모델 경로만 Qwen으로 바꾸시면 됩니다)
# -----------------------------
MAX_LEN = 1024
# Qwen2.5-0.5B-Instruct 또는 1.5B 경로로 설정하세요.
BASE_MODEL_DIR = os.path.expanduser("~/Desktop/models/Qwen2.5-0.5B-Instruct")
TRAIN_JSONL = "data/rola_data/train_oracle_plus_idk_ver2.jsonl"
OUT_DIR = "outputs/qwen2.5_0.5b_rag_lora"

# -----------------------------
# 2. 템플릿 일치화 (SmolLM2와 100% 동일)
# -----------------------------
def build_text(ex):
    p = ex["prompt"].rstrip()
    r = ex["response"].rstrip()
    c = (ex.get("context") or "").rstrip()

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
            "Answer:\n" 
            f"{r}"
        )
    else:
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
    # 토크나이저 로드 (Qwen은 trust_remote_code 권장)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 모델 로드 (Qwen은 bfloat16 지원이 매우 좋습니다)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    model.config.use_cache = False

    # 데이터셋 준비
    ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    ds = ds.map(build_text)

    # 미리 토큰화 (버전 에러 방지용)
    def tokenize_fn(batch):
        out = tok(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # 3. LoRA 설정 (Qwen 최적화)
    lora_cfg = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear", # Qwen2.5의 모든 선형 레이어 타겟팅
        bias="none",
    )

    # 4. 학습 설정 (Qwen은 조금 더 정교한 학습이 필요하여 Epoch과 LR 조정)
    sft_cfg = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,   # 모델이 커졌으므로 8에서 4로 조정
        gradient_accumulation_steps=8,  # 배치를 맞추기 위해 4에서 8로 조정
        learning_rate=1e-4,             # Qwen은 1e-4가 더 안정적입니다.
        num_train_epochs=3,             # RAG 규칙을 확실히 배우기 위해 3 Epoch 권장
        logging_steps=10,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
        eval_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=sft_cfg,
        processing_class=tok,
    )

    print(f"--- 🚀 Qwen2.5 RAG 학습 시작 (Target: {BASE_MODEL_DIR}) ---")
    trainer.train()
    
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"[OK] 저장 완료: {OUT_DIR}")

if __name__ == "__main__":
    main()