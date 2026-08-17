"""Shared Gemini client singleton and model name constants."""

import bootstrap
import os
import threading
from google import genai

GENERATION_MODEL = 'gemini-3.5-flash'
EMBEDDING_MODEL = 'gemini-embedding-2'
PDF_PARSER_MODEL = 'gemini-3.5-flash-lite'
HYDE_GENERATION_MODEL = 'gemini-3.5-flash-lite'

_client: genai.Client | None = None
_client_lock = threading.Lock()


def get_client() -> genai.Client:
    """Return a process-wide Gemini client, creating it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
                if not api_key:
                    raise EnvironmentError(
                        'No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.'
                    )
                _client = genai.Client(api_key=api_key)
    return _client
