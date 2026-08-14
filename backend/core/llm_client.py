"""Central Gemini API client for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client
