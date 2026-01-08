"""
Prompt Templates for SLM + RAG Experiments
==========================================

Modes:
1. ONLY_SLM      : No retrieval baseline
2. NAIVE_RAG     : BM25 / Vector naive RAG
3. FAIL_RESPONSE : Unified fallback
"""

# =====================
# 1. ONLY SLM (No-RAG)
# =====================

ONLY_SLM_PROMPT = """Answer the following question concisely.
If you do not know the answer, say "I don't know". 

Question:
{question}

Answer:
"""

# =====================
# 2. NAIVE RAG (BM25 / Vector)
# =====================

NAIVE_RAG_PROMPT = """Answer the question using ONLY the information in the context below.
If the answer cannot be found in the context, say "I don't know".
Do not use any external knowledge.

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
