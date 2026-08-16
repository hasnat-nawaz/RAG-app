import bootstrap
import asyncio
import html
import io
import re
import unicodedata
from pathlib import Path
from google.genai import types
from pypdf import PdfReader, PdfWriter
from llm_client import PDF_PARSER_MODEL, get_client
from models.schemas import DocumentLoadInput
_HTML_COMMENT = re.compile('<!--.*?-->', re.DOTALL)
_TOC_LEADERS = re.compile('[.\\u2026]{4,}')
_UNICODE_SPACE = re.compile('[\\u00a0\\u1680\\u2000-\\u200a\\u202f\\u205f\\u3000]')
_ZERO_WIDTH = re.compile('[\\u200b\\u200c\\u200d\\ufeff\\u00ad]')
_TRAILING_WS = re.compile('[ \\t]+$', re.MULTILINE)
_INTERNAL_MULTI_SPACE = re.compile('(?<=\\S) {2,}')
_MULTI_BLANK = re.compile('\\n{3,}')
_FENCE = re.compile('^```(?:markdown|md)?\\s*|\\s*```$', re.IGNORECASE)
PAGES_PER_CHUNK = 4
MAX_BATCH_REQUESTS = 15
MAX_CONCURRENT_PARSES = 5
MAX_OUTPUT_TOKENS = 65536
PARSE_PROMPT = 'Convert this PDF page batch into clean GitHub-flavored markdown, optimized for downstream chunking and embedding. This batch is a 4-page fragment of a larger document being processed in parallel — treat it as mid-stream, not standalone.\n\nOUTPUT FORMAT\n- Output markdown only — no preamble, no commentary, no wrapping code fences.\n- Keep all numbers, names, dates, and wording exact — do not paraphrase, summarize, translate, or correct the source text.\n- Preserve the original language(s) exactly as written, including mixed-language documents.\n\nHEADING DETECTION (use structure, not styling)\n- The primary signal for a heading is a numbering pattern at the start of a line (e.g. "5", "5.5", "5.5.4", "5.5.4.1") followed by a short title-like phrase — NOT visual cues like bold or font size. Map depth to numbering depth: "5" → #, "5.5" → ##, "5.5.4" → ###, "5.5.4.1" → ####.\n- This numbering-to-depth mapping is absolute and based only on counting the dot-separated numeric components — never on how large, bold, or "chapter-like" the heading looks on the page, and never on what depth the last heading you emitted was. A heading numbered "5.2" is always ## (two components), even if it starts a new page, starts a new batch, or is set in a large title font that makes it look like a new top-level chapter. Do not let visual prominence override the number. Example of the failure to avoid: emitting "# 5.2 Semi-Autonomous Fleet Control Task" as a top-level heading (#) just because it\'s large and bold — it must be "## 5.2 Semi-Autonomous Fleet Control Task" because it has one dot.\n- Apply this even if the heading is plain, unstyled text with no bold/large font — do not skip a heading just because it looks like a normal paragraph visually.\n- Never guess or fabricate a heading level for a section you can\'t see the full numbering for — output exactly the number shown, converted to the matching depth.\n\nUNNUMBERED SUBHEADINGS (narrow exception)\n- Only numbered or clearly standalone section titles are promoted to headings by default. Do not promote a bolded term, a table caption, or a run-in phrase (e.g. "**Herd Movement Mode:** description continues on the same line...") to a heading just because it\'s styled — leave those as inline bold text.\n- Narrow exception: if a short line with no numbering sits completely alone (its own line, no trailing colon, nothing else on that line) and is immediately followed by two or more paragraphs/definitions that clearly belong to it as a group, it is a real informal subheading and should be kept as a heading — but map it to ONE level deeper than the nearest numbered heading currently in scope, never to the same depth as a numbered heading. Example: inside a numbered "### 5.2.2 Job Description" section, an unnumbered standalone line "Herd Movement" that introduces several bolded sub-definitions becomes "#### Herd Movement" (one level past the enclosing ###), not "### Herd Movement". This keeps informal groupings from colliding with real numbered sections at the same header depth.\n\nSTRIP NOISE\n- Running headers/footers, page numbers, "Page X of Y" markers.\n- Logos, decorative images, watermarks, draft stamps — skip entirely, no placeholder, no description.\n- Boilerplate repeated identically across pages (confidentiality notices, letterhead, copyright footers).\n\nSTRIP NAVIGATIONAL / FRONT-MATTER BLOCKS\n- Detect and fully drop table-of-contents pages, indexes, lists of tables, lists of figures, lists of abbreviations, and "in this chapter" mini-outlines — in any language. These are recognizable by structure, not just by title: repeated short lines, often ending in a page number or dotted leader (e.g. "5.1 Dynamic Swarm Tasks .......... 12"), that mirror headings appearing later in the body.\n- Drop the entire block: title line AND all its entries. Do not keep the heading line ("# İçindekiler", "# TABLES", etc.) while stripping only the entries — the heading line itself is also noise, since the real section appears later in the body with real content.\n- Do not confuse this with a genuine body section that happens to contain a short list — the giveaway is page-number/leader formatting and 1:1 mirroring of headings elsewhere in the document.\n\nSTRUCTURE AND CLEANLINESS\n- Merge text broken across lines due to PDF layout: de-hyphenate split words, rejoin wrapped lines into full sentences/paragraphs.\n- Reconstruct correct reading order for multi-column layouts before output — never interleave columns.\n- Preserve true paragraph/section breaks only. Do not insert extra blank lines or fragmented single-line "paragraphs" caused by PDF layout.\n- Collapse redundant whitespace; exactly one blank line between block elements (headings, paragraphs, tables, lists).\n- Keep footnotes attached to their referencing content, or grouped at the end of the relevant section — never left as orphaned fragments.\n\nTABLES\n- Any grid of short aligned values — including small, borderless groupings like a version-history block ("V1.0 | 09.01.2026 | First Version") — is a table. Emit it as a proper GitHub-flavored markdown pipe table with a header row and alignment row, even if the source PDF renders it without visible gridlines. Do not leave it as run-on plain text just because it lacks ruled borders.\n- If a table continues across pages within this batch, do not repeat the header row on later pages (per CROSS-CHUNK CONTINUITY below). If a table\'s data rows start at the very top of page 1 with no header visible, emit only the data rows — don\'t fabricate a header.\n\nTABLE FIDELITY (overrides the general best-effort rule below, for table cells only)\n- Tables in this document type often carry scoring rules, dates, or other values where a wrong number is worse than a visible gap. If a specific table cell is genuinely illegible or ambiguous — not just structurally complex — output `[unclear]` in that cell instead of inventing a plausible-sounding value. Do not guess a number, a percentage, or a category label to make a row "read naturally" if you can\'t actually see it.\n- This exception applies only to individual table cells. For ordinary prose, continue to follow the EDGE CASES rule below (best-effort transcription, no inline flagging) — do not start adding `[unclear]` markers into running text.\n\nMULTI-LANGUAGE ADJACENCY\n- Some source documents interleave two language versions of the same content (e.g. an English line and its Turkish equivalent in the same list or table row). Preserve both — do not drop either — but never let them run together as one sentence with no separator. Keep each language version on its own line, its own list item, or its own table column. Never produce output like "...whilst maintaining the formation.Vformasyonuna geç" where a second language\'s text is glued directly onto the end of the first with no space or line break — insert a line break (or, inside a table, a separate column) between them instead.\n\nCROSS-CHUNK CONTINUITY (critical — output is concatenated with adjacent batches)\n- Never insert a title, heading, or "---" separator for the batch itself. This is a fragment, not a document.\n- If the top of page 1 of this batch clearly continues a sentence, paragraph, list, or table from before (starts lowercase mid-sentence, a table with no header row, a list item not starting at 1/a/i) — output it as a raw continuation: no heading, no blank-line separator, no restated context.\n- If the bottom of the last page is cut off mid-sentence, mid-list, or mid-table, output it exactly as far as it goes. Do not complete the sentence, close the list, or add closing punctuation you can\'t see.\n- Do not include any page/batch metadata ("Page 12", "Continued", "Batch 3 of 15") in the output.\n\nEDGE CASES\n- If a page is blank, or contains only stripped noise (headers/footers/logos/TOC entries), output nothing for that page — no placeholder, no note.\n- If prose content is ambiguous or partially illegible, give your best-effort transcription rather than omitting it or flagging uncertainty inline. (For table cells specifically, use the TABLE FIDELITY rule above instead — that rule wins for cells.)\n\nTreat the result as one continuous fragment of a single well-structured markdown document — no page-artifact residue, no chunk-boundary artifacts, nothing that would break when this output is concatenated with the batches before and after it.\n'
_loader: 'DocumentLoader | None' = None

class DocumentLoader:

    def __init__(self) -> None:
        self.client = get_client()

    @staticmethod
    def clean_markdown(md_text: str) -> str:
        text = md_text.replace('\r\n', '\n').replace('\r', '\n')
        text = unicodedata.normalize('NFC', text)
        text = html.unescape(text)
        text = _FENCE.sub('', text)
        text = _ZERO_WIDTH.sub('', text)
        text = _UNICODE_SPACE.sub(' ', text)
        text = _HTML_COMMENT.sub('', text)
        text = _TOC_LEADERS.sub(' ', text)
        text = _TRAILING_WS.sub('', text)
        text = _INTERNAL_MULTI_SPACE.sub(' ', text)
        text = _MULTI_BLANK.sub('\n\n', text)
        return text.strip()

    def split_pdf(self, file_path: Path) -> tuple[int, list[bytes]]:
        payload = DocumentLoadInput(file_path=file_path)
        reader = PdfReader(str(payload.file_path))
        if not reader.pages:
            raise RuntimeError(f'{payload.file_path.name} has no pages.')
        chunks: list[bytes] = []
        for start in range(0, len(reader.pages), PAGES_PER_CHUNK):
            writer = PdfWriter()
            for page in reader.pages[start:start + PAGES_PER_CHUNK]:
                writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            chunks.append(buffer.getvalue())
        return (len(reader.pages), chunks)

    def generate_markdown_piece(self, pdf_bytes: bytes, chunk_index: int) -> str:
        response = self.client.models.generate_content(model=PDF_PARSER_MODEL, contents=[types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'), PARSE_PROMPT], config=types.GenerateContentConfig(temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS))
        return (response.text or '').strip()

    async def aload_as_markdown(self, file_path: str | Path) -> str:
        from ingest_pipeline import collect_markdown_only
        return await collect_markdown_only(self, Path(file_path))

    def load_as_markdown(self, file_path: str | Path) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aload_as_markdown(file_path))
        raise RuntimeError('load_as_markdown() cannot run inside an active event loop; use: await document_loader.aload_as_markdown(path)')

def get_document_loader() -> DocumentLoader:
    global _loader
    if _loader is None:
        _loader = DocumentLoader()
    return _loader
