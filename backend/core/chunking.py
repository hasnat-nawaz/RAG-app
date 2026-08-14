import bootstrap  # noqa: F401

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from models.schemas import Chunk, ChunkMarkdownInput

HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def _document_to_chunk(doc: Document, source: str) -> Chunk:
    return Chunk(
        source=source,
        metadata=dict(doc.metadata),
        content=doc.page_content,
    )


def chunk_markdown(text: str, source: str) -> list[dict]:
    payload = ChunkMarkdownInput(text=text, source=source)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    chunks = splitter.split_text(payload.text)
    return [
        _document_to_chunk(chunk, payload.source).model_dump()
        for chunk in chunks
        if chunk.page_content.strip()
    ]
