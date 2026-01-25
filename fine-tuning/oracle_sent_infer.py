import json
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
from utility.inference import InferenceEngine


# -------------------------
# IO
# -------------------------
def load_examples(path: str, input_format: str) -> List[Dict[str, Any]]:
    """
    input_format:
      - jsonl: line-delimited JSON objects
      - json : a list OR {"data": [...]} 형태
    """
    p = Path(path)

    if input_format == "jsonl":
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    if isinstance(obj, list):
        return obj

    raise ValueError("Unsupported JSON structure. Provide a list or {data:[...]}.")


# -------------------------
# Oracle ctx utilities
# -------------------------
def build_sentence_map(documents_sentences: List[List[List[str]]]) -> Dict[str, str]:
    """
    documents_sentences:
      [
        [ ["0a","..."], ["0b","..."], ...],   # doc0
        [ ["1a","..."], ...],                 # doc1
      ]
    -> {"0a": "...", "0b": "...", ...}
    """
    mp: Dict[str, str] = {}
    for doc_sents in documents_sentences:
        for sid, sent in doc_sents:
            mp[sid] = sent
    return mp


def sort_sentence_keys(keys: List[str]) -> List[str]:
    """
    "4a", "4b", ... "4aa" 같은 키를 대충이라도 안정적으로 정렬.
    - 앞 숫자(doc idx) 기준
    - 그 다음 suffix 길이/사전순
    """
    def key_fn(k: str) -> Tuple[int, int, str]:
        num = 10**9
        i = 0
        while i < len(k) and k[i].isdigit():
            i += 1
        if i > 0:
            num = int(k[:i])
        suffix = k[i:]
        return (num, len(suffix), suffix)

    return sorted(keys, key=key_fn)


def oracle_ctx_sentences(
    ex: Dict[str, Any],
    keep_order: bool = True,
    max_sentences: Optional[int] = None,
) -> List[str]:
    """
    Sentence-Oracle:
      - all_relevant_sentence_keys에 해당하는 문장들을 contexts 리스트로 반환
    """
    rel_keys = ex.get("all_relevant_sentence_keys", [])
    doc_sents = ex.get("documents_sentences", [])
    if not rel_keys or not doc_sents:
        return []

    sent_map = build_sentence_map(doc_sents)

    keys = rel_keys
    if keep_order:
        keys = sort_sentence_keys(rel_keys)

    ctx: List[str] = []
    for k in keys:
        s = sent_map.get(k, "").strip()
        if s:
            ctx.append(s)

    if max_sentences is not None:
        ctx = ctx[:max_sentences]

    return ctx


def truncate_by_chars(ctxs: List[str], max_chars: int) -> List[str]:
    """
    SLM 안전장치: context가 너무 길면 앞에서부터 문자 수 기준으로 컷
    """
    out: List[str] = []
    total = 0
    for s in ctxs:
        if total + len(s) > max_chars:
            break
        out.append(s)
        total += len(s)
    return out


# -------------------------
# Main
# -------------------------
async def main(args):
    examples = load_examples(args.data_path, args.input_format)
    print(f"[INFO] loaded {len(examples)} examples")

    # InferenceEngine 내부에서 mode에 따라 프롬프트를 선택하는 구조라면
    # 여기 mode는 네가 이미 쓰던 그대로 둠.
    engine = InferenceEngine("NAIVE_RAG", args.model_name, verbose=False)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 샘플 수 제한(옵션)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
        print(f"[INFO] using first {len(examples)} examples (max_samples)")

    with out_path.open("w", encoding="utf-8") as out:
        for ex in tqdm(examples, desc=f"Infer[ORACLE_SENT:{args.model_name}]"):
            qid = ex.get("id", "")
            q = ex.get("question", "")

            # 1) Oracle ctx: relevant sentences만
            ctxs = oracle_ctx_sentences(
                ex,
                keep_order=True,
                max_sentences=args.max_sentences
            )

            # (옵션) context 길이 컷
            if args.max_ctx_chars is not None:
                ctxs = truncate_by_chars(ctxs, args.max_ctx_chars)

            # 2) Generate
            pred = await engine(q, ctxs)

            # 3) Save
            # - delucionqa 같이 GT가 없거나 애매하면 response를 reference로 쓰는 경우가 많음
            # - techqa 같이 GT가 있으면 answer/ground_truth를 쓰면 됨
            if args.use_reference == "response":
                reference_answer = ex.get("response", "")
            elif args.use_reference == "answer":
                reference_answer = ex.get("answer", ex.get("ground_truth", ""))
            else:  # "auto"
                # 우선순위: ground_truth -> answer -> response
                reference_answer = ex.get("ground_truth", ex.get("answer", ex.get("response", "")))

            rec = {
                "metadata": {
                    "id": qid,
                    "model": args.model_name,
                    "retrieval": "ORACLE_SENT",
                    "top_k": None,
                    "dataset_name": ex.get("dataset_name", None),
                },
                "question": q,
                "contexts": ctxs,
                "prediction": pred,

                # ✅ 비교/평가용 reference (GT 문자열이 아니라 response를 쓰고 싶으면 --use_reference response)
                "reference_answer": reference_answer,

                # ✅ boolean 근거 준수 여부(데이터에 있으면)
                "adherence_score": ex.get("adherence_score", None),

                # ✅ 분석용(선택)
                "all_relevant_sentence_keys": ex.get("all_relevant_sentence_keys", []),
                "all_utilized_sentence_keys": ex.get("all_utilized_sentence_keys", None),
                "unsupported_response_sentence_keys": ex.get("unsupported_response_sentence_keys", None),
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--input_format", type=str, choices=["json", "jsonl"], default="json")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)

    # Oracle sentence 옵션
    ap.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Oracle로 뽑은 relevant sentence를 최대 몇 개까지 줄지(옵션)"
    )

    # SLM 안전장치(옵션)
    ap.add_argument(
        "--max_ctx_chars",
        type=int,
        default=None,
        help="context 총 문자수 제한(옵션, SLM 폭주 방지)"
    )

    # 샘플 수 제한(옵션)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="처리할 샘플 수(디버깅/빠른 실험용). 예: 100"
    )

    # reference 선택
    ap.add_argument(
        "--use_reference",
        type=str,
        choices=["auto", "ground_truth", "answer", "response"],
        default="auto",
        help=(
            "reference_answer에 무엇을 저장할지 선택. "
            "auto: ground_truth->answer->response 우선순위, "
            "response: delucionqa류에 추천"
        )
    )

    args = ap.parse_args()
    asyncio.run(main(args))
