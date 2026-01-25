import json
import re
import random
from pathlib import Path

random.seed(42)

def flatten_docs_sentences(documents_sentences):
    # returns list of (sid, text)
    pairs = []
    for doc in documents_sentences:
        for sid, sent in doc:
            sent = sent.strip()
            if sent:
                pairs.append((sid, sent))
    return pairs

def build_context_from_keys(all_pairs, keep_keys, max_sents=40):
    keep = set(keep_keys)
    kept = [(sid, txt) for sid, txt in all_pairs if sid in keep]
    kept = kept[:max_sents]
    return "\n".join([f"[{sid}] {txt}" for sid, txt in kept])

def build_context_realistic(all_pairs, max_sents=40):
    # 그대로 앞에서부터 max_sents만
    kept = all_pairs[:max_sents]
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
Answer: <one sentence>
Evidence: [<sentence_ids>]
"""


def short_answer_from_response_sentences(ex):
    # 가장 간단/안전: response_sentences의 첫 문장(a)을 사용
    rs = ex.get("response_sentences", [])
    if rs and isinstance(rs, list) and len(rs) > 0:
        # ["a", "..."] 형태
        first = rs[0][1].strip()
        # "According to the context..." 같은 군더더기 제거(옵션)
        first = re.sub(r"^According to the context provided,\s*", "", first).strip()
        # 너무 길면 첫 문장만 유지 (이미 문장 단위라 보통 OK)
        return first

    # fallback: response의 첫 줄
    resp = (ex.get("response") or "").strip()
    return resp.split("\n")[0].strip() if resp else ""

def pick_best_evidence(ex):
    # 가장 깔끔: answer를 직접 지지하는 키가 있으면 그걸 사용
    # sentence_support_information에서 response_sentence_key == "a"의 supporting_sentence_keys 중
    # "general" 같은 특수값 제외하고 1개 선택
    ssi = ex.get("sentence_support_information", [])
    if isinstance(ssi, list):
        for item in ssi:
            if item.get("response_sentence_key") == "a":
                keys = item.get("supporting_sentence_keys", [])
                keys = [k for k in keys if k not in ("general", "supported_without_sentence",
                                                     "well_known_fact", "numerical_reasoning")]
                if keys:
                    return [keys[0]]

    # fallback: all_utilized_sentence_keys에서 1개
    u = ex.get("all_utilized_sentence_keys", [])
    if isinstance(u, list) and len(u) > 0:
        return [u[-1]]  # 보통 결론/정의 문장이 뒤에 있는 경우가 많아서 마지막
    return []

def make_cfail_context(all_pairs, remove_keys, max_sents=40):
    remove = set(remove_keys)
    kept = [(sid, txt) for sid, txt in all_pairs if sid not in remove]
    kept = kept[:max_sents]
    return "\n".join([f"[{sid}] {txt}" for sid, txt in kept])

def main(in_path, out_path, mode="realistic", add_cfail=True, cfail_ratio=1.0,
         max_ctx=40, oracle_use_relevant=True):
    """
    mode:
      - realistic: documents_sentences 그대로(앞에서부터 max_ctx)
      - oracle: all_relevant_sentence_keys만으로 context 구성
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_drop = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)

            # SFT 단계에서는 unsupported 섞인 건 드랍 권장
            if ex.get("unsupported_response_sentence_keys"):
                n_drop += 1
                continue

            q = ex.get("question", "").strip()
            docs_sent = ex.get("documents_sentences")
            if not q or not docs_sent:
                n_drop += 1
                continue

            all_pairs = flatten_docs_sentences(docs_sent)

            # context 구성
            if mode == "oracle":
                rel = ex.get("all_relevant_sentence_keys", [])
                if not rel:
                    n_drop += 1
                    continue
                context = build_context_from_keys(all_pairs, rel, max_sents=max_ctx)
            else:
                context = build_context_realistic(all_pairs, max_sents=max_ctx)

            prompt = build_prompt(q, context)

            ans = short_answer_from_response_sentences(ex)
            if not ans:
                n_drop += 1
                continue

            ev = pick_best_evidence(ex)
            # evidence 없으면(특수값만 있거나) utilized에서라도 뽑는 걸 권장
            if not ev:
                ev = ex.get("all_utilized_sentence_keys", [])[:1]

            response = f"Answer: {ans}\nEvidence: [{', '.join(ev)}]"

            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n_pos += 1

            # C-fail hard negative 추가
            if add_cfail and cfail_ratio > 0:
                # oracle remove set은 relevant, realistic remove set은 "best evidence" 중심으로 제거
                remove_keys = set(ex.get("all_relevant_sentence_keys", [])) if oracle_use_relevant else set(ev)
                cfail_ctx = make_cfail_context(all_pairs, remove_keys, max_sents=max_ctx)
                cfail_prompt = build_prompt(q, cfail_ctx)
                cfail_resp = 'Answer: I don\'t know.\nEvidence: []'

                k = int(round(cfail_ratio))
                for _ in range(k):
                    fout.write(json.dumps({"prompt": cfail_prompt, "response": cfail_resp}, ensure_ascii=False) + "\n")
                    n_neg += 1

    print(f"done: pos={n_pos}, cfail={n_neg}, dropped={n_drop}, out={out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl")
    ap.add_argument("--out", dest="out", required=True, help="output jsonl {prompt,response}")
    ap.add_argument("--mode", choices=["realistic", "oracle"], default="realistic")
    ap.add_argument("--max_ctx", type=int, default=40)
    ap.add_argument("--add_cfail", action="store_true")
    ap.add_argument("--cfail_ratio", type=float, default=1.0)
    ap.add_argument("--oracle_remove_relevant", action="store_true",
                    help="cfail 만들 때 all_relevant_sentence_keys를 제거(권장)")
    args = ap.parse_args()

    main(
        in_path=args.inp,
        out_path=args.out,
        mode=args.mode,
        add_cfail=args.add_cfail,
        cfail_ratio=args.cfail_ratio,
        max_ctx=args.max_ctx,
        oracle_use_relevant=args.oracle_remove_relevant,
    )
