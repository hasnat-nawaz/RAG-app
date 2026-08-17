"""LanceDB vector store with hybrid search, HyDE, and FTS support."""

import bootstrap
import json
import math
from pathlib import Path
from threading import Lock
import lancedb
from lancedb.index import FTS
from embedding import get_embedder
from models.schemas import AddRecordsInput, DEFAULT_TOP_K, KeywordSearchInput, RetrievedDocument, VectorSearchInput
from pipeline_log import log
from query_optimization.hypothetical_document import generate_hypothetical_document
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / 'storage' / 'lanceDB'
DEFAULT_TABLE_NAME = 'EmbeddingsTable'
_store: 'VectorStore | None' = None

def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)

def encode_metadata(metadata: dict | None) -> str:
    """Serialize chunk metadata to a JSON string for LanceDB storage."""
    clean: dict = {}
    for key, value in (metadata or {}).items():
        if value is None or _is_nan(value):
            continue
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except Exception:
                pass
        clean[str(key)] = value
    return json.dumps(clean, ensure_ascii=False)

def decode_metadata(value) -> dict:
    """Parse stored metadata back into a plain dict."""
    if value is None or _is_nan(value):
        return {}
    if isinstance(value, dict):
        return {str(k): None if v is None or _is_nan(v) else v for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

class VectorStore:
    """Read and write embedded document chunks in LanceDB."""

    def __init__(self, db_path: str | Path=DEFAULT_DB_PATH, table_name: str=DEFAULT_TABLE_NAME, *, embedder=None) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.embedder = embedder or get_embedder()
        self.db = lancedb.connect(self.db_path)
        self.table = self._open_table_if_exists()
        self._fts_index_ready: bool | None = None
        self._write_lock = Lock()
        if self.table is not None and self._metadata_column_is_struct():
            with self._write_lock:
                self._migrate_metadata_to_json_unlocked()

    def _open_table_if_exists(self):
        if self.table_name in self.db.list_tables().tables:
            return self.db.open_table(self.table_name)
        return None

    def ensure_queryable(self) -> None:
        """Verify the database exists and has at least one row."""
        if self.table is None:
            self.table = self._open_table_if_exists()
        if self.table is None:
            raise RuntimeError('Database is empty.')
        try:
            row_count = self.table.count_rows()
        except Exception as exc:
            if self._looks_like_missing_or_corrupt(exc):
                try:
                    self.reset_table()
                except Exception:
                    pass
                raise RuntimeError('Database is empty.') from exc
            raise RuntimeError('Something went wrong while searching the database.') from exc
        if row_count == 0:
            raise RuntimeError('Database is empty.')

    @staticmethod
    def _looks_like_missing_or_corrupt(exc: BaseException) -> bool:
        text = str(exc).lower()
        return 'not found' in text or 'no such file' in text or '.lance' in text or ('fragment' in text)

    def reset_table(self) -> None:
        with self._write_lock:
            self._drop_table_unlocked()

    def _drop_table_unlocked(self) -> None:
        if self.table_name in self.db.list_tables().tables:
            self.db.drop_table(self.table_name)
        self.table = None
        self._fts_index_ready = None

    def _next_id(self) -> int:
        if self.table is None:
            return 1
        try:
            if self.table.count_rows() == 0:
                return 1
            max_id = int(self.table.to_pandas()['id'].max())
            return max_id + 1
        except Exception:
            self._drop_table_unlocked()
            return 1

    @staticmethod
    def _strip_embedding_text(chunk: dict) -> dict:
        data = {k: v for k, v in chunk.items() if k != 'embedding_text'}
        metadata = dict(data.get('metadata') or {})
        metadata.pop('embedding_text', None)
        data['metadata'] = metadata
        return data

    @staticmethod
    def _sql_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def delete_by_source(self, source: str) -> int:
        if not source:
            return 0
        with self._write_lock:
            return self._delete_by_source_unlocked(source)

    def count_by_source(self, source: str) -> int:
        if not source:
            return 0
        with self._write_lock:
            if self.table is None:
                self.table = self._open_table_if_exists()
            if self.table is None:
                return 0
            try:
                df = self.table.to_pandas()
                if df is None or getattr(df, 'empty', True):
                    return 0
                if 'source' not in df.columns:
                    return 0
                return int((df['source'] == source).sum())
            except Exception:
                return 0

    def has_source(self, source: str) -> bool:
        return self.count_by_source(source) > 0

    def _delete_by_source_unlocked(self, source: str) -> int:
        if self.table is None:
            self.table = self._open_table_if_exists()
        if self.table is None:
            return 0
        try:
            before = int(self.table.count_rows())
            if before == 0:
                return 0
            self.table.delete(f'source = {self._sql_string(source)}')
            after = int(self.table.count_rows())
            deleted = max(0, before - after)
            if after == 0:
                self._drop_table_unlocked()
            elif deleted:
                try:
                    self._update_fts_index()
                except Exception:
                    pass
            return deleted
        except Exception:
            return 0

    def _metadata_column_is_struct(self) -> bool:
        if self.table is None:
            return False
        try:
            schema = self.table.schema
            for field in schema:
                if getattr(field, 'name', None) == 'metadata':
                    type_name = str(getattr(field, 'type', '')).lower()
                    return type_name.startswith('struct')
            return False
        except Exception:
            return False

    def _migrate_metadata_to_json_unlocked(self) -> None:
        if self.table is None:
            return
        try:
            raw_rows = self.table.to_pandas().to_dict(orient='records')
        except Exception as exc:
            raise RuntimeError(f'Failed to export rows for metadata migration: {exc}') from exc
        migrated: list[dict] = []
        for row in raw_rows:
            meta = decode_metadata(row.get('metadata'))
            vector = row.get('vector')
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            migrated.append({'id': int(row['id']), 'source': str(row['source']), 'content': str(row['content']), 'metadata': encode_metadata(meta), 'vector': list(vector)})
        self._drop_table_unlocked()
        if migrated:
            self.table = self.db.create_table(self.table_name, data=migrated)
            self._fts_index_ready = None
            self._update_fts_index()

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        sanitized = [self._strip_embedding_text(chunk) if isinstance(chunk, dict) else chunk for chunk in chunks]
        payload = AddRecordsInput(chunks=sanitized, embeddings=embeddings)
        sources = sorted({chunk.source for chunk in payload.chunks})
        with self._write_lock:
            if self.table is not None and self._metadata_column_is_struct():
                self._migrate_metadata_to_json_unlocked()
            start_id = self._next_id()
            rows = [{'id': row_id, 'source': chunk.source, 'content': chunk.content, 'metadata': encode_metadata(chunk.metadata), 'vector': list(embedding)} for row_id, (chunk, embedding) in enumerate(zip(payload.chunks, payload.embeddings), start=start_id)]
            try:
                if self.table is None:
                    self.table = self.db.create_table(self.table_name, data=rows)
                else:
                    self.table.add(rows)
            except Exception as exc:
                text = str(exc).lower()
                if 'does not exist in table schema' in text or 'field' in text:
                    try:
                        self._migrate_metadata_to_json_unlocked()
                        if self.table is None:
                            self.table = self.db.create_table(self.table_name, data=rows)
                        else:
                            self.table.add(rows)
                    except Exception as retry_exc:
                        for source in sources:
                            self._delete_by_source_unlocked(source)
                        raise RuntimeError(f'Failed to write embeddings to the database: {retry_exc}') from retry_exc
                else:
                    for source in sources:
                        self._delete_by_source_unlocked(source)
                    raise RuntimeError(f'Failed to write embeddings to the database: {exc}') from exc
            self._update_fts_index()
            return len(rows)

    def _has_fts_index(self) -> bool:
        if self._fts_index_ready is not None:
            return self._fts_index_ready
        if self.table is None:
            return False
        for index in self.table.list_indices():
            columns = list(getattr(index, 'columns', None) or [])
            name = str(getattr(index, 'name', '')).lower()
            if 'content' in columns or 'content' in name or 'fts' in name:
                self._fts_index_ready = True
                return True
        self._fts_index_ready = False
        return False

    def _update_fts_index(self) -> None:
        if self._has_fts_index():
            self.table.optimize()
            return
        self._create_fts_index()

    def _create_fts_index(self) -> None:
        self.table.create_index('content', config=FTS(), replace=True)
        self._fts_index_ready = True

    def _rows_to_documents(self, rows: list[dict]) -> list[dict]:
        documents: list[dict] = []
        for row in rows:
            data = dict(row)
            data['metadata'] = decode_metadata(data.get('metadata'))
            documents.append(RetrievedDocument.model_validate(data).model_dump(by_alias=True))
        return documents

    @staticmethod
    def merge_documents(*doc_lists: list[dict]) -> list[dict]:
        merged: list[dict] = []
        seen: set[int] = set()
        for docs in doc_lists:
            for doc in docs:
                doc_id = doc.get('id')
                if doc_id is not None:
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                merged.append(doc)
        return merged

    def semantic_search(self, query_vector: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Find the top-k chunks closest to a query embedding (cosine similarity)."""
        if self.table is None:
            raise RuntimeError('Database is empty.')
        payload = VectorSearchInput(query_vector=query_vector, top_k=top_k)
        try:
            rows = self.table.search(payload.query_vector).metric('cosine').limit(payload.top_k).to_list()
        except Exception as exc:
            if self._looks_like_missing_or_corrupt(exc):
                try:
                    self.reset_table()
                except Exception:
                    pass
                raise RuntimeError('Database is empty.') from exc
            raise RuntimeError('Something went wrong while searching the database.') from exc
        return self._rows_to_documents(rows)

    def bm25(self, query: str, top_k: int=DEFAULT_TOP_K) -> list[dict]:
        if self.table is None:
            raise RuntimeError('Database is empty.')
        payload = KeywordSearchInput(query=query, top_k=top_k)
        try:
            if not self._has_fts_index():
                self._create_fts_index()
            rows = self.table.search(payload.query, query_type='fts', fts_columns='content').limit(payload.top_k).to_list()
        except Exception as exc:
            if self._looks_like_missing_or_corrupt(exc):
                try:
                    self.reset_table()
                except Exception:
                    pass
                raise RuntimeError('Database is empty.') from exc
            raise RuntimeError('Something went wrong while searching the database.') from exc
        return self._rows_to_documents(rows)

    def hyde(self, query: str, top_k: int=DEFAULT_TOP_K) -> list[dict]:
        """Retrieve chunks by embedding a HyDE hypothetical answer passage."""
        payload = KeywordSearchInput(query=query, top_k=top_k)
        log("HYDE", "generating hypothetical document")
        hypothetical_doc = generate_hypothetical_document(payload.query)
        log("HYDE", "embedding hypothetical document")
        query_vector = self.embedder.embed_query(hypothetical_doc)
        docs = self.semantic_search(query_vector, top_k=payload.top_k)
        log("HYDE", f"found {len(docs)} chunks")
        return docs

def get_vector_store(db_path: str | Path=DEFAULT_DB_PATH, table_name: str=DEFAULT_TABLE_NAME, *, embedder=None) -> VectorStore:
    """Return the shared VectorStore instance for the given path."""
    global _store
    resolved = Path(db_path)
    if _store is None or _store.db_path != resolved or _store.table_name != table_name:
        _store = VectorStore(db_path=resolved, table_name=table_name, embedder=embedder)
    elif embedder is not None:
        _store.embedder = embedder
    return _store
