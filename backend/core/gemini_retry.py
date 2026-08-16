from __future__ import annotations

import asyncio
import re
import time
from typing import Callable, TypeVar

from logutil import plog

MAX_RETRIES = 5
QUOTA_COOLDOWN_SECONDS = 62.0
DEFAULT_TRANSIENT_WAIT_SECONDS = 8.0

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_FIELD = re.compile(
    r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?",
    re.IGNORECASE,
)

T = TypeVar("T")


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    return " ".join(parts)


def is_quota_error(exc: BaseException) -> bool:
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


def parse_retry_seconds(exc: BaseException) -> float | None:
    text = _error_text(exc)
    match = _RETRY_IN.search(text)
    if match:
        return max(1.0, float(match.group(1)))
    match = _RETRY_DELAY_FIELD.search(text)
    if match:
        return max(1.0, float(match.group(1)))
    return None


def wait_seconds_for_error(exc: BaseException) -> float:
    explicit = parse_retry_seconds(exc)
    if explicit is not None:
        return explicit
    if is_quota_error(exc):
        return QUOTA_COOLDOWN_SECONDS
    return DEFAULT_TRANSIENT_WAIT_SECONDS


def should_retry(exc: BaseException) -> bool:
    return is_quota_error(exc) or is_transient_server_error(exc)


def run_with_retries(
    label: str,
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    pipeline: str = "retry",
) -> T:
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
            delay = wait_seconds_for_error(exc)
            plog(
                pipeline,
                event="retry_wait",
                label=label,
                kind="quota" if is_quota_error(exc) else "transient",
                attempt=f"{attempt}/{max_retries}",
                wait_s=delay,
                error=str(exc)[:160],
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{label}: failed after {max_retries} attempts. Last error: {last_error}"
    )


async def arun_with_retries(
    label: str,
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    pipeline: str = "retry",
) -> T:
    return await asyncio.to_thread(
        run_with_retries,
        label,
        fn,
        max_retries=max_retries,
        pipeline=pipeline,
    )
