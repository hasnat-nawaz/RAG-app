"""Heading-aware Markdown chunking for the RAG pipeline.

Embeddings see only section body text. Heading hierarchy is stored in metadata.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


# Map Markdown heading markers to metadata keys the splitter will populate.
HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


# Serialize a Document to content + metadata for the saved JSON.
def _document_to_dict(doc: Document) -> dict:
    return {
        "metadata": doc.metadata,
        "content": doc.page_content,    
    }


# Split Markdown on H1–H6; hierarchy lives in metadata, never in content.
def chunk_markdown(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("chunk_markdown expects a non-empty Markdown string.")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    chunks = splitter.split_text(text)

    # Drop header-only slices, then return content + metadata dicts.
    return [_document_to_dict(chunk) for chunk in chunks if chunk.page_content.strip()]


# Local test run: load converted Markdown, chunk it, and write the objects to JSON.
if __name__ == "__main__":
    storage_dir = Path(__file__).resolve().parents[1] / "storage"
    input_dir = storage_dir / "output_texts"
    output_dir = storage_dir / "output_texts"
    docs = ["swarm.md"]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            markdown_path = input_dir / doc
            if not markdown_path.is_file():
                raise FileNotFoundError(f"Markdown not found: {markdown_path}")

            print(f"Chunking '{doc}' with MarkdownHeaderTextSplitter...")
            markdown_text = markdown_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(markdown_text)

            output_file_path = output_dir / f"{Path(doc).stem}.chunks.json"
            output_file_path.write_text(
                json.dumps(chunks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Saved {len(chunks)} chunk(s) to: {output_file_path}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
