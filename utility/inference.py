"""
inference.py
============

Inference logic for SLM + RAG experiments.

Modes:
- NO_PROMPT
- ONLY_SLM
- NAIVE_RAG

Responsibilities:
- prompt construction
- LLM invocation
- NO retrieval, NO evaluation
"""

from typing import List, Optional
from .prompt import ONLY_SLM_PROMPT, NAIVE_RAG_PROMPT, FAIL_RESPONSE
from .ollama import generate


class InferenceEngine:
    def __init__(
        self,
        mode: str,
        model: str = "smollm2:135m",
        verbose: bool = False,
        max_context_chars: int = 4000,
    ):
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.max_context_chars = max_context_chars

        if mode == "NO_PROMPT":
            self._infer_fn = self._infer_no_prompt
        elif mode == "ONLY_SLM":
            self._infer_fn = self._infer_only_slm
        elif mode == "NAIVE_RAG":
            self._infer_fn = self._infer_naive_rag
        else:
            raise ValueError(f"Unknown inference mode: {mode}")

    # =====================
    # public interface
    # =====================
    async def __call__(
        self,
        question: str,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> str:
        return await self._infer_fn(question, retrieved_chunks)

    # =====================
    # inference implementations
    # =====================
    async def _infer_no_prompt(
        self,
        question: str,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> str:
        """
        Raw inference:
        - No instructions
        - No formatting
        - Question (+ optional context) only
        """
        parts = []

        if retrieved_chunks:
            context = "\n\n".join(retrieved_chunks)
            context = context[: self.max_context_chars]
            parts.append(context)

        parts.append(question)

        prompt = "\n\n".join(parts)

        answer = await generate(
            prompt,
            model=self.model,
            verbose=self.verbose,
        )
        return answer.strip() if answer else FAIL_RESPONSE

    async def _infer_only_slm(
        self,
        question: str,
        _: Optional[List[str]] = None,
    ) -> str:
        prompt = ONLY_SLM_PROMPT.format(question=question)

        answer = await generate(
            prompt,
            model=self.model,
            verbose=self.verbose,
        )
        return answer.strip() if answer else FAIL_RESPONSE

    async def _infer_naive_rag(
        self,
        question: str,
        retrieved_chunks: Optional[List[str]],
    ) -> str:
        if not retrieved_chunks:
            return FAIL_RESPONSE

        context = "\n\n".join(retrieved_chunks)
        context = context[: self.max_context_chars]

        prompt = NAIVE_RAG_PROMPT.format(
            context=context,
            question=question,
        )

        answer = await generate(
            prompt,
            model=self.model,
            verbose=self.verbose,
        )
        return answer.strip() if answer else FAIL_RESPONSE
