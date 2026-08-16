import bootstrap
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from models.schemas import ChunkMarkdownInput, EmbeddableChunk
HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3'), ('####', 'Header 4'), ('#####', 'Header 5'), ('######', 'Header 6')]
_HEADER_KEYS: list[str] = [label for _, label in HEADERS_TO_SPLIT_ON]
_chunker: 'Chunker | None' = None

class Chunker:

    def __init__(self, max_chunk_chars: int=1800, chunk_overlap_chars: int=200, min_chunk_chars: int=150, merge_small_sections: bool=True, prepend_header_path_to_embedding_text: bool=True) -> None:
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=True)
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.merge_small_sections = merge_small_sections
        self.prepend_header_path = prepend_header_path_to_embedding_text
        self._prose_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_chars, chunk_overlap=chunk_overlap_chars, separators=['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ', ''])

    def chunk_markdown(self, text: str, source: str) -> list[dict]:
        payload = ChunkMarkdownInput(text=text, source=source)
        raw_docs = [doc for doc in self.splitter.split_text(payload.text) if doc.page_content.strip()]
        sections = self._merge_small_leaf_sections(raw_docs) if self.merge_small_sections else raw_docs
        chunks: list[dict] = []
        for section_index, doc in enumerate(sections):
            header_path = self._build_header_path(doc.metadata)
            pieces = self._split_section(doc.page_content)
            total = len(pieces)
            for piece_index, piece_text in enumerate(pieces):
                metadata = dict(doc.metadata)
                metadata['header_path'] = header_path
                metadata['section_index'] = section_index
                metadata['chunk_index'] = piece_index
                metadata['chunk_count'] = total
                metadata['char_count'] = len(piece_text)
                metadata['approx_tokens'] = max(1, len(piece_text) // 4)
                metadata['chunk_id'] = self._make_chunk_id(source, header_path, section_index, piece_index)
                embedding_text = f'{header_path}\n\n{piece_text}' if self.prepend_header_path and header_path else piece_text
                chunks.append(EmbeddableChunk(source=source, metadata=metadata, content=piece_text, embedding_text=embedding_text).model_dump())
        return chunks

    @staticmethod
    def _build_header_path(metadata: dict) -> str:
        parts = [metadata[key] for key in _HEADER_KEYS if metadata.get(key)]
        return ' > '.join(parts)

    @staticmethod
    def _make_chunk_id(source: str, header_path: str, section_index: int, piece_index: int) -> str:
        raw = f'{source}::{header_path}::{section_index}::{piece_index}'
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    def _merge_small_leaf_sections(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        merged: list[Document] = []
        i = 0
        while i < len(docs):
            doc = docs[i]
            content = doc.page_content.strip()
            if len(content) < self.min_chunk_chars and i + 1 < len(docs) and self._is_header_descendant(doc.metadata, docs[i + 1].metadata):
                nxt = docs[i + 1]
                combined = f'{content}\n\n{nxt.page_content.strip()}'
                merged.append(Document(page_content=combined, metadata=nxt.metadata))
                i += 2
                continue
            merged.append(doc)
            i += 1
        return merged

    @staticmethod
    def _is_header_descendant(parent_meta: dict, child_meta: dict) -> bool:
        for key in _HEADER_KEYS:
            parent_val = parent_meta.get(key)
            if parent_val is None:
                continue
            if child_meta.get(key) != parent_val:
                return False
        parent_depth = sum((1 for k in _HEADER_KEYS if parent_meta.get(k)))
        child_depth = sum((1 for k in _HEADER_KEYS if child_meta.get(k)))
        return child_depth > parent_depth

    def _split_section(self, content: str) -> list[str]:
        content = content.strip()
        if not content:
            return []
        if len(content) <= self.max_chunk_chars:
            return [content]
        blocks = self._split_into_blocks(content)
        return self._pack_blocks(blocks)

    @staticmethod
    def _split_into_blocks(text: str) -> list[str]:
        lines = text.split('\n')
        blocks: list[str] = []
        buf: list[str] = []
        in_table = False

        def flush() -> None:
            if buf:
                joined = '\n'.join(buf).strip('\n')
                if joined.strip():
                    blocks.append(joined)
                buf.clear()
        for line in lines:
            is_table_row = line.lstrip().startswith('|')
            if line.strip() == '':
                if not in_table:
                    flush()
                continue
            if is_table_row and (not in_table):
                flush()
                in_table = True
            elif not is_table_row and in_table:
                flush()
                in_table = False
            buf.append(line)
        flush()
        return blocks

    def _pack_blocks(self, blocks: list[str]) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        current_len = 0

        def block_len(b: str) -> int:
            return len(b) + 2
        for block in blocks:
            if len(block) > self.max_chunk_chars:
                if current:
                    packed.append('\n\n'.join(current))
                    current, current_len = ([], 0)
                packed.extend(self._split_oversized_block(block))
                continue
            projected = current_len + block_len(block)
            if current and projected > self.max_chunk_chars:
                packed.append('\n\n'.join(current))
                current = self._select_overlap_blocks(current)
                current_len = sum((block_len(b) for b in current))
            current.append(block)
            current_len += block_len(block)
        if current:
            packed.append('\n\n'.join(current))
        return packed

    def _select_overlap_blocks(self, window_blocks: list[str]) -> list[str]:
        if self.chunk_overlap_chars <= 0 or not window_blocks:
            return []
        overlap: list[str] = []
        used = 0
        for block in reversed(window_blocks):
            if self._is_table_block(block):
                break
            if used + len(block) > self.chunk_overlap_chars:
                break
            overlap.insert(0, block)
            used += len(block)
        return overlap

    @staticmethod
    def _is_table_block(block: str) -> bool:
        stripped = block.strip()
        first_line = stripped.splitlines()[0] if stripped else ''
        return first_line.lstrip().startswith('|')

    def _split_oversized_block(self, block: str) -> list[str]:
        if self._is_table_block(block):
            return self._split_oversized_table(block)
        return self._prose_splitter.split_text(block)

    def _split_oversized_table(self, block: str) -> list[str]:
        rows = block.strip('\n').split('\n')
        if not rows:
            return [block]
        is_alignment_row = len(rows) > 1 and set(rows[1].replace('|', '').strip()) <= {'-', ':', ' '}
        header_rows = rows[:2] if is_alignment_row else rows[:1]
        data_rows = rows[len(header_rows):]
        header_block = '\n'.join(header_rows)
        chunks: list[str] = []
        current = [header_block]
        current_len = len(header_block)
        for row in data_rows:
            if current_len + len(row) + 1 > self.max_chunk_chars and len(current) > 1:
                chunks.append('\n'.join(current))
                current = [header_block, row]
                current_len = len(header_block) + len(row)
            else:
                current.append(row)
                current_len += len(row) + 1
        if len(current) > 1:
            chunks.append('\n'.join(current))
        elif not chunks:
            chunks.append(header_block)
        return chunks

def get_chunker() -> Chunker:
    global _chunker
    if _chunker is None:
        _chunker = Chunker()
    return _chunker
