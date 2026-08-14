import bootstrap  # noqa: F401

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

import torch
import torch._dynamo
import torch._logging

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._logging.set_logs(dynamo=logging.ERROR, graph_breaks=False)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.disable_progress_bar()

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

from models.schemas import DocumentLoadInput

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TOC_LEADERS = re.compile(r"[.\u2026]{4,}")
_UNICODE_SPACE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_INTERNAL_MULTI_SPACE = re.compile(r"(?<=\S) {2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")

_loader: "DocumentLoader | None" = None


def create_warmup_pdf() -> io.BytesIO:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


class DocumentLoader:
    def __init__(self, *, do_ocr: bool = False):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.warmup()

    def warmup(self) -> bool:
        print("warming up layout models...")
        warmup_start = time.perf_counter()
        dummy_stream = DocumentStream(name="warmup.pdf", stream=create_warmup_pdf())
        self.converter.convert(dummy_stream)
        print(f"warmup time taken = {time.perf_counter() - warmup_start:.1f}s")
        return True

    @staticmethod
    def clean_markdown(md_text: str) -> str:
        text = md_text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text)
        text = html.unescape(text)
        text = _ZERO_WIDTH.sub("", text)
        text = _UNICODE_SPACE.sub(" ", text)
        text = _HTML_COMMENT.sub("", text)
        text = _TOC_LEADERS.sub(" ", text)
        text = _TRAILING_WS.sub("", text)
        text = _INTERNAL_MULTI_SPACE.sub(" ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        return text.strip()

    def load_as_markdown(self, file_path: str | Path) -> str:
        payload = DocumentLoadInput(file_path=file_path)
        result = self.converter.convert(payload.file_path)
        markdown = result.document.export_to_markdown(
            escape_html=False,
            escape_underscores=False,
            image_placeholder="",
            compact_tables=True,
        )
        return self.clean_markdown(markdown)


def get_document_loader(*, do_ocr: bool = False) -> DocumentLoader:
    global _loader
    if _loader is None:
        _loader = DocumentLoader(do_ocr=do_ocr)
    return _loader

