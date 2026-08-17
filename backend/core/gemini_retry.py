"""Retry helpers for transient Gemini API failures (quota and server errors)."""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Callable, TypeVar

MAX_RETRIES = 5
COOLDOWN_SECONDS = 65.0
TRANSIENT_WAIT_SECONDS = 8.0

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)

T = TypeVar("T")


def _error_text(exc: BaseException) -> str:
    """Collect message text from an exception and its common API attributes."""
    parts = [str(exc)]
    for attr in ("message", "details"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def is_quota_error(exc: BaseException) -> bool:
    """Return True when the error looks like a rate-limit or quota exhaustion."""
    text = _error_text(exc).lower()
    code = getattr(exc, "code", None)
    return (
        code == 429
        or "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "rate-limit" in text
        or "ratelimit" in text
    )


def is_transient_server_error(exc: BaseException) -> bool:
    """Return True for temporary server-side failures worth retrying."""
    text = _error_text(exc).lower()
    code = getattr(exc, "code", None)
    return (
        code in (500, 503, 504)
        or "503" in text
        or "unavailable" in text
        or "deadline" in text
        or "timeout" in text
        or "temporarily" in text
        or "high demand" in text
    )


def should_retry(exc: BaseException) -> bool:
    """Return True when the exception is eligible for a retry."""
    return is_quota_error(exc) or is_transient_server_error(exc)


def _wait_for_error(exc: BaseException) -> float:
    """Compute backoff seconds before the next retry attempt."""
    if is_quota_error(exc):
        return COOLDOWN_SECONDS + random.uniform(0.5, 2.5)
    m = _RETRY_IN.search(_error_text(exc))
    if m:
        return max(1.0, float(m.group(1))) + random.uniform(0.2, 1.0)
    return TRANSIENT_WAIT_SECONDS + random.uniform(0.2, 1.0)


def run_with_retries(
    label: str,
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
) -> T:
    """Run fn with retries on transient API errors; raise after max_retries."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if not should_retry(exc) or attempt >= max_retries:
                raise RuntimeError(
                    f"{label}: failed after {attempt} attempt(s): {exc}"
                ) from exc
            time.sleep(_wait_for_error(exc))
    raise RuntimeError(
        f"{label}: failed after {max_retries} attempts. Last error: {last_error}"
    )


async def arun_with_retries(
    label: str,
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
) -> T:
    """Async wrapper that runs run_with_retries in a thread pool."""
    return await asyncio.to_thread(
        run_with_retries,
        label,
        fn,
        max_retries=max_retries,
    )
