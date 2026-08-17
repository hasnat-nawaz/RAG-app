#!/usr/bin/env python3
"""Pre-download the local reranker model during setup."""

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent
CORE_DIR = BACKEND_DIR / 'core'
for path in (BACKEND_DIR, CORE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import bootstrap
from reranker import MODEL_NAME, Reranker


def main() -> None:
    """Load the cross-encoder once so subsequent runs use the cached weights."""
    print(f'==> Caching reranker model ({MODEL_NAME})')
    Reranker()
    print('==> Local models ready')


if __name__ == '__main__':
    main()
