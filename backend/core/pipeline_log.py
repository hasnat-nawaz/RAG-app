"""One-line tagged logger for upload and query pipeline stages."""

_TAG_WIDTH = 10


def log(tag: str, message: str) -> None:
    """Print a single tagged log line, e.g. [QUERY] retrieving — methods: hybrid."""
    padded = f"[{tag}]".ljust(_TAG_WIDTH)
    print(f"{padded} {message}", flush=True)
