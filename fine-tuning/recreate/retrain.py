import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model


IGNORE_INDEX = -100


def pick_lora_targets(model) -> List[str]:
    """
    SmolLM2 계열은 보통 attention proj에 q_proj/k_proj/v_proj/o_proj가 존재.
    모델 구조가 다른 경우를 대비해 존재하는 모듈명만 골라줌.
    """
    # 후보들(많이 쓰는 패턴)
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    names = set()
    for n, _ in model.named_modules():
        # 모듈 path의 마지막 토큰을 확인
        last = n.split(".")[-1]
        if last in candidates:
            names.add(last)

    if names:
        return sorted(list(names))

    # 혹시 다른 네이밍(예: query_key_value 등)일 경우를 대비한 fallback
    fallback = ["query_key_value", "dense", "fc1", "fc2"]
    names = set()
    for n, _ in model.named_modules():
        last = n.split(".")[-1]
        if last in fallback:
            names.add(last)

    return sorted(list(names)) if names else candidates  # 마지막 fallback


@dataclass
class CausalSFTCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        features에는 이미 input_ids/attention_mask/labels가 들어있음.
        여기서 배치 내 최대 길이 기준으로 패딩만 수행.
        """
        pad_id = self.tokenizer.pad_token_id
        max_len = min(self.max_length, max(len(f["input_ids"]) for f in features))

        def pad_list(lst, pad_val):
            if len(lst) >= max_len:
                return lst[:max_len]
            return lst + [pad_val] * (max_len - len(lst))

        input_ids = [pad_list(f["input_ids"], pad_id) for f in features]
        attention_mask = [pad_list(f["attention_mask"], 0) for f in features]
        labels = [pad_list(f["labels"], IGNORE_INDEX) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_example(tokenizer, prompt: str, response: str, max_length: int) -> Dict[str, Any]:
    """
    핵심: prompt는 loss 제외, response만 loss 적용.
    labels에서 prompt 토큰은 IGNORE_INDEX(-100)로 마스킹.
    """
    # special token은 prompt/response에 이미 포함됐을 수 있으니 add_special_tokens=False
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # EOS는 response 끝에 붙여주는 게 보통 안정적
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + resp_ids + [eos]
    attention_mask = [1] * len(input_ids)

    labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [eos]

    # 너무 길면 뒤를 자르되, prompt 일부만 남고 response가 날아가는 건 최악이므로
    # "prompt를 먼저 자르는" 전략을 취함.
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length

        # prompt가 충분히 길다면 prompt 앞부분부터 잘라냄
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
            # 다시 결합
            input_ids = prompt_ids + resp_ids + [eos]
            attention_mask = [1] * len(input_ids)
            labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [eos]
        else:
            # 그래도 넘치면 response도 잘릴 수밖에 없음(데이터/길이 조정 권장)
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            labels = labels[-max_length:]

            # 잘린 구간에서 prompt/response 경계가 깨졌을 수 있으니
            # 안전하게 "labels 중 앞쪽 연속 IGNORE_INDEX"를 늘리는 방식은 생략(복잡도↑).
            # 실전에서는 max_length를 늘리거나 prompt를 줄이는 게 좋음.

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--train_file", type=str, required=True, help="jsonl with {prompt,response}")
    ap.add_argument("--output_dir", type=str, default="adapter_smollm2_135m")
    ap.add_argument("--max_length", type=int, default=2048)

    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Train
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=0, help="0이면 epoch 저장")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # 안정성/메모리
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # checkpointing 시 cache 끄는 게 일반적

    # LoRA targets 자동 선택
    target_modules = pick_lora_targets(model)
    print(f"[LoRA] target_modules = {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # dataset load
    ds = load_dataset("json", data_files={"train": args.train_file})["train"]

    # map -> tokenized samples
    def map_fn(ex):
        prompt = ex["prompt"]
        response = ex["response"]
        return build_example(tokenizer, prompt, response, args.max_length)

    ds = ds.map(map_fn, remove_columns=ds.column_names)

    collator = CausalSFTCollator(tokenizer=tokenizer, max_length=args.max_length)

    # TrainingArguments
    # save_steps=0이면 epoch 기반 저장으로 유도
    save_strategy = "steps" if args.save_steps and args.save_steps > 0 else "epoch"

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else 0,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    # ✅ 핵심: adapter만 저장 (Ollama ADAPTER로 쓰기 위함)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 저장 결과 확인용 안내
    print("\n[OK] Saved LoRA adapter to:", args.output_dir)
    print("Expect files like:")
    print(f"  {args.output_dir}/adapter_model.safetensors")
    print(f"  {args.output_dir}/adapter_config.json")


if __name__ == "__main__":
    main()
