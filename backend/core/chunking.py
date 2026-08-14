"""Heading-aware Markdown chunking for the RAG pipeline.

Embeddings see only section body text. Heading hierarchy is stored in metadata.
"""

import bootstrap  # noqa: F401

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def _document_to_dict(doc: Document, source: str) -> dict:
    return {
        "source": source,
        "metadata": doc.metadata,
        "content": doc.page_content,
    }


def chunk_markdown(text: str, source: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("chunk_markdown expects a non-empty Markdown string.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("chunk_markdown expects a non-empty source filename.")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    chunks = splitter.split_text(text)

    return [
        _document_to_dict(chunk, source)
        for chunk in chunks
        if chunk.page_content.strip()
    ]
