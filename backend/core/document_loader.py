# Silence noisy third-party logs before those libraries are imported.
import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# Quiet Torch Dynamo so graph-break warnings never hit the terminal.
import torch
import torch._dynamo
import torch._logging

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._logging.set_logs(dynamo=logging.ERROR, graph_breaks=False)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.disable_progress_bar()

# Standard library and PDF/Markdown conversion imports.
import html
import io
import re
import time
import unicodedata
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocumentStream
from pypdf import PdfWriter


# Precompiled cleanup patterns so they are not rebuilt on every document.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TOC_LEADERS = re.compile(r"[.\u2026]{4,}")          # "....." / "……" TOC fillers
_UNICODE_SPACE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_INTERNAL_MULTI_SPACE = re.compile(r"(?<=\S) {2,}")  # keep list indentation
_MULTI_BLANK = re.compile(r"\n{3,}")


# Build a tiny in-memory PDF used only to warm up Docling's layout models.
def create_warmup_pdf() -> io.BytesIO:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


class DocumentLoader:
    # Configure Docling and run one dummy conversion so models are loaded once.
    def __init__(self, *, do_ocr: bool = False):
        print("1. Configuring pipeline options...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

        # PDF-only converter with the options above.
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Warmup pass so the first real document is not paying model-load cost.
        print("2. Warming up layout models in memory...")
        warmup_start = time.perf_counter()
        dummy_stream = DocumentStream(name="warmup.pdf", stream=create_warmup_pdf())
        self.converter.convert(dummy_stream)
        print(f"3. Models loaded in {time.perf_counter() - warmup_start:.2f} seconds!")

    # Strip Docling artifacts so the Markdown is stable input for chunking.
    @staticmethod
    def clean_markdown(md_text: str) -> str:
        # Normalize line endings, unicode, and leftover HTML entities.
        text = md_text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text)
        text = html.unescape(text)

        # Remove invisible chars, HTML comments, and TOC dotted leaders.
        text = _ZERO_WIDTH.sub("", text)
        text = _UNICODE_SPACE.sub(" ", text)
        text = _HTML_COMMENT.sub("", text)
        text = _TOC_LEADERS.sub(" ", text)

        # Collapse extra whitespace without flattening nested list indentation.
        text = _TRAILING_WS.sub("", text)
        text = _INTERNAL_MULTI_SPACE.sub(" ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        return text.strip()

    # Convert a PDF on disk to cleaned Markdown.
    def load_as_markdown(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        result = self.converter.convert(path)
        markdown = result.document.export_to_markdown(
            escape_html=False,
            escape_underscores=False,
            image_placeholder="",
            compact_tables=True,
        )
        return self.clean_markdown(markdown)


# Local test run: convert listed PDFs into storage/output_texts.
if __name__ == "__main__":
    print("Initializing DocumentLoader...")
    loader = DocumentLoader()

    storage_dir = Path(__file__).resolve().parents[1] / "storage"
    input_dir = storage_dir / "uploaded_docs"
    output_dir = storage_dir / "output_texts"
    docs = ["swarm.pdf"]

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            start_time = time.perf_counter()
            output_file_path = output_dir / f"{Path(doc).stem}.md"
            print(f"Converting '{doc}' to Markdown...")
            markdown_output = loader.load_as_markdown(input_dir / doc)
            output_file_path.write_text(markdown_output, encoding="utf-8")
            print(f"Successfully saved extracted text to: {output_file_path}")
            print(f"Time Taken : {time.perf_counter() - start_time:.2f}s")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
