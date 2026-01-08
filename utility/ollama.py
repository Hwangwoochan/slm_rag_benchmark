"""
ollama.py
=========

Lightweight Ollama interface for SLM inference.
This module ONLY handles text generation (LLM I/O).
"""

import time
import ollama


async def generate(
    prompt: str,
    model: str = "smollm2:135m",
    temperature: float = 0.0,
    verbose: bool = False,
) -> str:
    """
    Generate text from Ollama model.
    """
    start = time.time()

    response = await ollama.AsyncClient().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )

    output = response["message"]["content"].strip()

    if verbose:
        duration = time.time() - start
        token_count = response.get("eval_count", 0)
        tps = token_count / duration if duration > 0 else 0.0

        print("=" * 40)
        print(f"Model: {model}")
        print(f"Latency: {duration:.3f}s")
        print(f"Tokens: {token_count}")
        print(f"Tokens/sec: {tps:.2f}")
        print("=" * 40)

    return output
