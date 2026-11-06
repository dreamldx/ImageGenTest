"""Timing log for outbound LLM / image-generation requests."""

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def log_request(kind: str, provider: str, model: str) -> Iterator[None]:
    """Print one line saying which `provider`/`model` a `kind` of request
    (e.g. ``llm``, ``image.generate``) used and how long it took, whether
    it succeeded or raised. Works around awaits in async code too."""
    start = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        print(f"[{kind}] {provider} model={model} failed after {time.perf_counter() - start:.2f}s: {exc!r}")
        raise
    print(f"[{kind}] {provider} model={model} took {time.perf_counter() - start:.2f}s")
