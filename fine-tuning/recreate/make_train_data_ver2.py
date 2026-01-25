import json
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

random.seed(42)

# -----------------------------
# Helpers: context/prompt
# -----------------------------
def flatten_docs_sentences(documents_sentences) -> List[Tuple[str, str]]:
    pairs = []
    for doc in documents_sentences:
        for sid, sent in doc:
            sent = (sent or "").strip()
            if sent:
                pairs.append((str(sid), sent))
    return pairs

def build_context_from_keys(all_pairs: List[Tuple[str, str]], keep_keys: List[str], max_sents: int) -> str:
    keep = set(map(str, keep_keys))
    kept = [(sid, txt) for sid, txt in all_pairs if sid in keep]
    kept = kept[:max_sents]
    return "\n".join([f"[{sid}] {txt}" for sid, txt in kept])

def build_context_realistic(all_pairs: List[Tuple[str, str]], max_sents: int) -> str:
    kept = all_pairs[:max_sents]
    return "\n".join([f"[{sid}] {txt}" for sid, txt in kept])

def make_cfail_context(all_pairs: List[Tuple[str, str]], remove_keys: List[str], max_sents: int) -> str:
    remove = set(map(str, remove_keys))
    kept = [(sid, txt) for sid, txt in all_pairs if sid not in remove]
    kept = kept[:max_sents]
    return "\n".join([f"[{sid}] {txt}" for sid, txt in kept])

def build_prompt(question: str, context: str) -> str:
    return f"""You are a careful assistant.

Rules:
- Use ONLY the provided context.
- Do NOT use external knowledge or assumptions.
- Answer concisely (1–3 sentences).
- If the answer cannot be determined from the context, say exactly: "I don't know".

Task:
Answer the question using only the given context.

Question:
{question}

Context:
{context}

Output format:
Answer: <1–3 sentences>
Evidence: [<sentence_ids>]
"""

# -----------------------------
# Helpers: teacher answer/evidence
# -----------------------------
_SPECIAL_EVIDENCE = {
    "general",
    "supported_without_sentence",
    "well_known_fact",
    "numerical_reasoning",
}

def _clean_answer(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^According to the context provided,\s*", "", t).strip()

    # 숫자만 / "1." 같은 잘림 방지
    if re.fullmatch(r"\d+\.?", t):
        return ""

    # "steps: 1." / "follow these steps: 1." 류 잘림 방지
    if re.search(r":\s*\d+\.?$", t):
        return ""
    if re.search(r"\bfollow these steps\b.*\d+\.?$", t, flags=re.IGNORECASE):
        return ""
    if re.search(r"\bstep(s)?\b.*\d+\.?$", t, flags=re.IGNORECASE):
        return ""

    # 너무 짧으면 제거(선택: 잡음 감소)
    if len(t) < 3:
        return ""

    return t

def get_teacher_answer(ex: Dict[str, Any], max_sentences: int = 3) -> str:
    """
    response_sentences(a,b,c...) 있으면 최대 max_sentences까지 이어붙임.
    없으면 response fallback.
    """
    rs = ex.get("response_sentences", [])
    parts = []
    if isinstance(rs, list) and rs:
        for key, sent in rs:
            if len(parts) >= max_sentences:
                break
            s = _clean_answer(sent)
            if s:
                parts.append(s)
        if parts:
            return " ".join(parts).strip()

    resp = _clean_answer(ex.get("response", ""))
    return resp

def get_supporting_keys_for_response(ex: Dict[str, Any], max_keys: int = 5) -> List[str]:
    """
    delucionqa는 fully_supported가 null인 경우가 많음.
    => sentence_support_information의 supporting_sentence_keys를 그대로 모아서 evidence로 사용.
    """
    ssi = ex.get("sentence_support_information", [])
    keys: List[str] = []
    if isinstance(ssi, list):
        for item in ssi:
            supp = item.get("supporting_sentence_keys", []) or []
            for k in supp:
                ks = str(k)
                if ks in _SPECIAL_EVIDENCE:
                    continue
                keys.append(ks)

    # 중복 제거(순서 유지)
    dedup = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup[:max_keys]

# -----------------------------
# Quality filter (delucionqa 맞춤)
# -----------------------------
def _get_float(ex: Dict[str, Any], key: str) -> Optional[float]:
    v = ex.get(key, None)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def is_high_quality_delucionqa(
    ex: Dict[str, Any],
    require_adherence_true: bool = True,
    min_relevance: Optional[float] = None,
    min_utilization: Optional[float] = None,
    min_completeness: Optional[float] = None,
) -> bool:
    # 1) 명시적으로 unsupported 있으면 탈락
    if ex.get("unsupported_response_sentence_keys"):
        return False

    # 2) adherence_score가 있으면(대부분 bool) 그걸 가장 신뢰
    #    - 없으면 스킵(데이터셋/스플릿에 따라 없을 수 있음)
    if require_adherence_true and "adherence_score" in ex:
        if ex.get("adherence_score") is not True:
            return False

    # 3) supporting_sentence_keys 존재 여부가 핵심 (fully_supported는 무시)
    ssi = ex.get("sentence_support_information", [])
    has_support = False
    if isinstance(ssi, list):
        for item in ssi:
            supp = item.get("supporting_sentence_keys", []) or []
            # special evidence만 있는 경우도 방지
            supp = [str(k) for k in supp if str(k) not in _SPECIAL_EVIDENCE]
            if len(supp) > 0:
                has_support = True
                break
    if not has_support:
        return False

    # 4) 점수 3종은 "있으면 적용" (없으면 통과)
    if min_relevance is not None:
        v = _get_float(ex, "relevance_score")
        if v is not None and v < min_relevance:
            return False

    if min_utilization is not None:
        v = _get_float(ex, "utilization_score")
        if v is not None and v < min_utilization:
            return False

    if min_completeness is not None:
        v = _get_float(ex, "completeness_score")
        if v is not None and v < min_completeness:
            return False

    return True

# -----------------------------
# Main
# -----------------------------
def main(
    in_path: str,
    out_path: str,
    mode: str = "oracle",             # oracle or realistic (positive context 구성)
    max_ctx: int = 40,
    teacher_max_sentences: int = 3,
    evidence_max_keys: int = 5,
    # delucionqa quality
    require_adherence_true: bool = True,
    min_relevance: Optional[float] = None,
    min_utilization: Optional[float] = None,
    min_completeness: Optional[float] = None,
    # IDK generation
    add_idk: bool = True,
    idk_ratio: float = 0.3,
    idk_remove_strategy: str = "supporting+utilized+relevant",
    idk_resp: str = 'Answer: I don\'t know.\nEvidence: []',
):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_pos = 0
    n_idk = 0
    n_drop = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            ex = json.loads(line)

            q = (ex.get("question") or "").strip()
            docs_sent = ex.get("documents_sentences")
            if not q or not docs_sent:
                n_drop += 1
                continue

            if not is_high_quality_delucionqa(
                ex,
                require_adherence_true=require_adherence_true,
                min_relevance=min_relevance,
                min_utilization=min_utilization,
                min_completeness=min_completeness,
            ):
                n_drop += 1
                continue

            all_pairs = flatten_docs_sentences(docs_sent)

            # ----- positive context -----
            if mode == "oracle":
                rel = ex.get("all_relevant_sentence_keys", []) or []
                if not rel:
                    n_drop += 1
                    continue
                context = build_context_from_keys(all_pairs, rel, max_sents=max_ctx)
            else:
                context = build_context_realistic(all_pairs, max_sents=max_ctx)

            prompt = build_prompt(q, context)

            # ----- teacher answer/evidence -----
            ans = get_teacher_answer(ex, max_sentences=teacher_max_sentences)
            if not ans:
                n_drop += 1
                continue
            if ans.strip() == "I don't know":
                n_drop += 1
                continue

            ev = get_supporting_keys_for_response(ex, max_keys=evidence_max_keys)

            if not ev:
                # fallback: utilized
                u = ex.get("all_utilized_sentence_keys", []) or []
                ev = [str(x) for x in u[: min(1, evidence_max_keys)]]

            response = f"Answer: {ans}\nEvidence: [{', '.join(ev)}]"
            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n_pos += 1

            # ----- IDK(c-fail) generation -----
            if add_idk and idk_ratio > 0:
                def should_add_once() -> bool:
                    if idk_ratio >= 1:
                        return True
                    return random.random() < idk_ratio

                repeats = int(idk_ratio) if idk_ratio >= 1 else 1
                for _ in range(repeats):
                    if not should_add_once():
                        continue

                    remove_keys = set()

                    if "supporting" in idk_remove_strategy:
                        remove_keys.update(ev)

                    if "utilized" in idk_remove_strategy:
                        remove_keys.update(map(str, ex.get("all_utilized_sentence_keys", []) or []))

                    if "relevant" in idk_remove_strategy:
                        remove_keys.update(map(str, ex.get("all_relevant_sentence_keys", []) or []))

                    if not remove_keys:
                        remove_keys.update(ev)

                    cfail_ctx = make_cfail_context(all_pairs, list(remove_keys), max_sents=max_ctx)
                    cfail_prompt = build_prompt(q, cfail_ctx)

                    fout.write(json.dumps({"prompt": cfail_prompt, "response": idk_resp}, ensure_ascii=False) + "\n")
                    n_idk += 1

    print(f"done: total={n_total}, pos={n_pos}, idk={n_idk}, dropped={n_drop}, out={out_path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl")
    ap.add_argument("--out", dest="out", required=True, help="output jsonl {prompt,response}")

    ap.add_argument("--mode", choices=["oracle", "realistic"], default="oracle")
    ap.add_argument("--max_ctx", type=int, default=40)

    ap.add_argument("--teacher_max_sentences", type=int, default=3)
    ap.add_argument("--evidence_max_keys", type=int, default=5)

    # delucionqa quality switches
    ap.add_argument("--no_require_adherence_true", action="store_true",
                    help="if set, do NOT require adherence_score==True (if field exists).")
    ap.add_argument("--min_relevance", type=float, default=None)
    ap.add_argument("--min_utilization", type=float, default=None)
    ap.add_argument("--min_completeness", type=float, default=None)

    # IDK
    ap.add_argument("--no_idk", action="store_true")
    ap.add_argument("--idk_ratio", type=float, default=0.3)
    ap.add_argument("--idk_remove_strategy", type=str, default="supporting+utilized+relevant")

    args = ap.parse_args()

    main(
        in_path=args.inp,
        out_path=args.out,
        mode=args.mode,
        max_ctx=args.max_ctx,
        teacher_max_sentences=args.teacher_max_sentences,
        evidence_max_keys=args.evidence_max_keys,
        require_adherence_true=not args.no_require_adherence_true,
        min_relevance=args.min_relevance,
        min_utilization=args.min_utilization,
        min_completeness=args.min_completeness,
        add_idk=not args.no_idk,
        idk_ratio=args.idk_ratio,
        idk_remove_strategy=args.idk_remove_strategy,
    )
