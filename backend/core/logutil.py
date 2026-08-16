from __future__ import annotations

import time
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value >= 100:
            return f"{value:.1f}"
        if value >= 10:
            return f"{value:.2f}"
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\n", " ").strip()
    if len(text) > 120:
        return text[:117] + "..."
    return text


def plog(pipeline: str, **fields: Any) -> None:
    parts = [f"[{pipeline}]"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_fmt(value)}")
    print(" ".join(parts), flush=True)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
