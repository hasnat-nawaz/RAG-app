import bootstrap  # noqa: F401

import asyncio
import html
import io
import re
import time
import unicodedata
from pathlib import Path

from google.genai import types
from pypdf import PdfReader, PdfWriter

from llm_client import PDF_PARSER_MODEL, get_client
from models.schemas import DocumentLoadInput

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TOC_LEADERS = re.compile(r"[.\u2026]{4,}")
_UNICODE_SPACE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_INTERNAL_MULTI_SPACE = re.compile(r"(?<=\S) {2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")
_FENCE = re.compile(r"^```(?:markdown|md)?\s*|\s*```$", re.IGNORECASE)

PAGES_PER_CHUNK = 4
MAX_BATCH_REQUESTS = 15
BATCH_COOLDOWN_SECONDS = 60
MAX_OUTPUT_TOKENS = 65536

PARSE_PROMPT = """\
Convert this PDF page batch into clean GitHub-flavored markdown, optimized for \
downstream chunking and embedding. This batch is a 4-page fragment of a larger \
document being processed in parallel — treat it as mid-stream, not standalone.

OUTPUT FORMAT
- Output markdown only — no preamble, no commentary, no wrapping code fences.
- Keep all numbers, names, dates, and wording exact — do not paraphrase, \
summarize, translate, or correct the source text.
- Preserve the original language(s) exactly as written, including mixed-language \
documents.

HEADING DETECTION (use structure, not styling)
- The primary signal for a heading is a numbering pattern at the start of a line \
(e.g. "5", "5.5", "5.5.4", "5.5.4.1") followed by a short title-like phrase — NOT \
visual cues like bold or font size. Map depth to numbering depth: "5" → #, "5.5" \
→ ##, "5.5.4" → ###, "5.5.4.1" → ####.
- Apply this even if the heading is plain, unstyled text with no bold/large font — \
do not skip a heading just because it looks like a normal paragraph visually.
- Do not promote a bolded term, a table caption, or a run-in phrase to a heading \
just because it's styled — only numbered or clearly standalone section titles \
qualify.
- Never guess or fabricate a heading level for a section you can't see the full \
numbering for — output exactly the number shown, converted to the matching depth.

STRIP NOISE
- Running headers/footers, page numbers, "Page X of Y" markers.
- Logos, decorative images, watermarks, draft stamps — skip entirely, no \
placeholder, no description.
- Boilerplate repeated identically across pages (confidentiality notices, \
letterhead, copyright footers).

STRIP NAVIGATIONAL / FRONT-MATTER BLOCKS
- Detect and fully drop table-of-contents pages, indexes, lists of tables, lists \
of figures, lists of abbreviations, and "in this chapter" mini-outlines — in any \
language. These are recognizable by structure, not just by title: repeated short \
lines, often ending in a page number or dotted leader (e.g. "5.1 Dynamic Swarm \
Tasks .......... 12"), that mirror headings appearing later in the body.
- Drop the entire block: title line AND all its entries. Do not keep the heading \
line ("# İçindekiler", "# TABLES", etc.) while stripping only the entries — the \
heading line itself is also noise, since the real section appears later in the \
body with real content.
- Do not confuse this with a genuine body section that happens to contain a short \
list — the giveaway is page-number/leader formatting and 1:1 mirroring of \
headings elsewhere in the document.

STRUCTURE AND CLEANLINESS
- Merge text broken across lines due to PDF layout: de-hyphenate split words, \
rejoin wrapped lines into full sentences/paragraphs.
- Reconstruct correct reading order for multi-column layouts before output — \
never interleave columns.
- Preserve true paragraph/section breaks only. Do not insert extra blank lines or \
fragmented single-line "paragraphs" caused by PDF layout.
- Collapse redundant whitespace; exactly one blank line between block elements \
(headings, paragraphs, tables, lists).
- Keep footnotes attached to their referencing content, or grouped at the end of \
the relevant section — never left as orphaned fragments.

CROSS-CHUNK CONTINUITY (critical — output is concatenated with adjacent batches)
- Never insert a title, heading, or "---" separator for the batch itself. This is \
a fragment, not a document.
- If the top of page 1 of this batch clearly continues a sentence, paragraph, \
list, or table from before (starts lowercase mid-sentence, a table with no header \
row, a list item not starting at 1/a/i) — output it as a raw continuation: no \
heading, no blank-line separator, no restated context.
- If the bottom of the last page is cut off mid-sentence, mid-list, or mid-table, \
output it exactly as far as it goes. Do not complete the sentence, close the \
list, or add closing punctuation you can't see.
- If a table continues across pages within this batch, do not repeat the header \
row on later pages. If a table's data rows start at the very top of page 1 with \
no header visible, emit only the data rows — don't fabricate a header.
- Do not include any page/batch metadata ("Page 12", "Continued", "Batch 3 of \
15") in the output.

EDGE CASES
- If a page is blank, or contains only stripped noise (headers/footers/logos/TOC \
entries), output nothing for that page — no placeholder, no note.
- If content is ambiguous or partially illegible, give your best-effort \
transcription rather than omitting it or flagging uncertainty inline.

Treat the result as one continuous fragment of a single well-structured markdown \
document — no page-artifact residue, no chunk-boundary artifacts, nothing that \
would break when this output is concatenated with the batches before and after it.
"""

_STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
_UPLOADED_DOCS_DIR = _STORAGE_DIR / "uploaded_docs"
_MARKDOWN_DOCS_DIR = _STORAGE_DIR / "markdown_docs"

_loader: "DocumentLoader | None" = None


class DocumentLoader:
    def __init__(self) -> None:
        self.client = get_client()

    @staticmethod
    def clean_markdown(md_text: str) -> str:
        text = md_text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text)
        text = html.unescape(text)
        text = _FENCE.sub("", text)
        text = _ZERO_WIDTH.sub("", text)
        text = _UNICODE_SPACE.sub(" ", text)
        text = _HTML_COMMENT.sub("", text)
        text = _TOC_LEADERS.sub(" ", text)
        text = _TRAILING_WS.sub("", text)
        text = _INTERNAL_MULTI_SPACE.sub(" ", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        return text.strip()

    def _split_pdf(self, file_path: Path) -> tuple[int, list[bytes]]:
        reader = PdfReader(str(file_path))
        if not reader.pages:
            raise RuntimeError(f"{file_path.name} has no pages.")

        chunks: list[bytes] = []
        for start in range(0, len(reader.pages), PAGES_PER_CHUNK):
            writer = PdfWriter()
            for page in reader.pages[start : start + PAGES_PER_CHUNK]:
                writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            chunks.append(buffer.getvalue())
        return len(reader.pages), chunks

    def _generate_chunk(self, pdf_bytes: bytes, chunk_index: int) -> str:
        response = self.client.models.generate_content(
            model=PDF_PARSER_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                PARSE_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        text = (response.text or "").strip()
        # Blank / noise-only pages may correctly return empty markdown.
        return text

    async def _parse_chunk(self, pdf_bytes: bytes, chunk_index: int) -> str:
        return await asyncio.to_thread(self._generate_chunk, pdf_bytes, chunk_index)

    async def _parse_chunks(self, chunks: list[bytes]) -> str:
        parts: list[str] = []
        total_batches = (len(chunks) + MAX_BATCH_REQUESTS - 1) // MAX_BATCH_REQUESTS

        for batch_number, start in enumerate(
            range(0, len(chunks), MAX_BATCH_REQUESTS),
            start=1,
        ):
            if batch_number > 1:
                print(
                    f"waiting {BATCH_COOLDOWN_SECONDS}s before batch "
                    f"{batch_number}/{total_batches}..."
                )
                await asyncio.sleep(BATCH_COOLDOWN_SECONDS)

            batch = chunks[start : start + MAX_BATCH_REQUESTS]
            print(
                f"sending batch {batch_number}/{total_batches}: "
                f"{len(batch)} parallel request(s)"
            )
            batch_parts = await asyncio.gather(
                *[
                    self._parse_chunk(chunk, start + offset)
                    for offset, chunk in enumerate(batch)
                ]
            )
            parts.extend(batch_parts)

        return "\n\n".join(part for part in parts if part)

    async def aload_as_markdown(self, file_path: str | Path) -> str:
        payload = DocumentLoadInput(file_path=file_path)
        page_count, chunks = self._split_pdf(payload.file_path)
        print(
            f"{payload.file_path.name}: {page_count} pages → "
            f"{len(chunks)} chunk(s), "
            f"max {MAX_BATCH_REQUESTS} request(s) at a time"
        )
        markdown = await self._parse_chunks(chunks)
        cleaned = self.clean_markdown(markdown)
        if not cleaned:
            raise RuntimeError(
                f"Document loader produced empty markdown for {payload.file_path.name}."
            )
        return cleaned

    def load_as_markdown(self, file_path: str | Path) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aload_as_markdown(file_path))
        raise RuntimeError(
            "load_as_markdown() cannot run inside an active event loop; "
            "use: await document_loader.aload_as_markdown(path)"
        )


def get_document_loader() -> DocumentLoader:
    global _loader
    if _loader is None:
        _loader = DocumentLoader()
    return _loader
