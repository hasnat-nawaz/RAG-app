from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from typing import Callable, TypeVar

from logutil import plog

MAX_RETRIES = 8
QUOTA_COOLDOWN_SECONDS = 65.0
DEFAULT_TRANSIENT_WAIT_SECONDS = 8.0

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_FIELD = re.compile(
    r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?",
    re.IGNORECASE,
)

T = TypeVar("T")

_cooldown_lock = threading.Lock()
_call_gate = threading.Lock()
_cooldown_until = 0.0
_serialize_until = 0.0


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    details = getattr(exc, "details", None)
    if details:
        parts.append(str(details))
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
    if is_quota_error(exc):
        base = explicit if explicit is not None else QUOTA_COOLDOWN_SECONDS
        return max(base, QUOTA_COOLDOWN_SECONDS) + random.uniform(0.5, 2.5)
    if explicit is not None:
        return explicit + random.uniform(0.2, 1.0)
    return DEFAULT_TRANSIENT_WAIT_SECONDS + random.uniform(0.2, 1.0)


def should_retry(exc: BaseException) -> bool:
    return is_quota_error(exc) or is_transient_server_error(exc)


def _wait_shared_cooldown(*, pipeline: str, label: str) -> None:
    while True:
        with _cooldown_lock:
            remaining = _cooldown_until - time.monotonic()
        if remaining <= 0:
            return
        plog(
            pipeline,
            event="cooldown_wait",
            label=label,
            wait_s=round(remaining, 2),
        )
        time.sleep(min(remaining, 5.0))


def _extend_cooldown(seconds: float, *, serialize: bool) -> None:
    global _cooldown_until, _serialize_until
    with _cooldown_lock:
        now = time.monotonic()
        _cooldown_until = max(_cooldown_until, now + seconds)
        if serialize:
            _serialize_until = max(_serialize_until, _cooldown_until + 3.0)


def _should_serialize() -> bool:
    with _cooldown_lock:
        return time.monotonic() < _serialize_until


def run_with_retries(
    label: str,
    fn: Callable[[], T],
    *,
    max_retries: int = MAX_RETRIES,
    pipeline: str = "retry",
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _wait_shared_cooldown(pipeline=pipeline, label=label)
        held_gate = False
        if _should_serialize():
            _call_gate.acquire()
            held_gate = True
        delay = 0.0
        try:
            _wait_shared_cooldown(pipeline=pipeline, label=label)
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if not should_retry(exc) or attempt >= max_retries:
                    raise RuntimeError(
                        f"{label}: failed after {attempt} attempt(s): {exc}"
                    ) from exc
                delay = wait_seconds_for_error(exc)
                quota = is_quota_error(exc)
                _extend_cooldown(delay, serialize=quota)
                plog(
                    pipeline,
                    event="retry_wait",
                    label=label,
                    kind="quota" if quota else "transient",
                    attempt=f"{attempt}/{max_retries}",
                    wait_s=round(delay, 2),
                    error=str(exc)[:160],
                )
        finally:
            if held_gate:
                _call_gate.release()
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
