"""
Prompt Templates for SLM + RAG Experiments
==========================================

Modes:
1. NO_PROMPT     : Raw model behavior (no instruction)
2. ONLY_SLM      : No retrieval baseline
3. NAIVE_RAG     : BM25 / Vector naive RAG
4. FAIL_RESPONSE : Unified fallback
"""

# =====================
# 0. NO PROMPT (RAW)
# =====================

NO_PROMPT = "{input}"

# =====================
# 1. ONLY SLM (No-RAG)
# =====================

ONLY_SLM_PROMPT = """Answer the following question.

Rules:
- Answer concisely (1–3 sentences).
- Be factual.
- If you do not know the answer, say exactly: "I don't know".

Question:
{question}

Answer:
"""

# =====================
# 2. NAIVE RAG (BM25 / Vector)
# =====================

NAIVE_RAG_PROMPT = """Answer the question using ONLY the information in the context below.

Rules:
- Use only the provided context.
- Do NOT use external knowledge or assumptions.
- Answer concisely (1–3 sentences).
- If the answer cannot be determined from the context, say exactly: "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

# =====================
# 3. Fallback / Failure
# =====================

FAIL_RESPONSE = "I don't know"
