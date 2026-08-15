import bootstrap  # noqa: F401

import asyncio
from pathlib import Path
from threading import Lock

import lancedb
from lancedb.index import FTS

from embedding import get_embedder
from models.schemas import (
    AddRecordsInput,
    DEFAULT_TOP_K,
    HybridSearchInput,
    KeywordSearchInput,
    RetrievedDocument,
    VectorRecord,
    VectorSearchInput,
)
from query_optimization.hypothetical_document import generate_hypothetical_document

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "lanceDB"
DEFAULT_TABLE_NAME = "EmbeddingsTable"

_store: "VectorStore | None" = None


class VectorStore:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        table_name: str = DEFAULT_TABLE_NAME,
        *,
        embedder=None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.embedder = embedder or get_embedder()
        self.db = lancedb.connect(self.db_path)
        self.table = self._open_table_if_exists()
        self._fts_index_ready: bool | None = None
        self._write_lock = Lock()

    def _open_table_if_exists(self):
        if self.table_name in self.db.list_tables().tables:
            return self.db.open_table(self.table_name)
        return None

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
            # count_rows()+1 collides after deletes; always allocate past max(id).
            max_id = int(self.table.to_pandas()["id"].max())
            return max_id + 1
        except Exception as exc:
            print(f"Vector store unreadable while allocating ids ({exc}); resetting table.")
            self._drop_table_unlocked()
            return 1

    @staticmethod
    def _strip_embedding_text(chunk: dict) -> dict:
        data = {k: v for k, v in chunk.items() if k != "embedding_text"}
        metadata = dict(data.get("metadata") or {})
        metadata.pop("embedding_text", None)
        data["metadata"] = metadata
        return data

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        sanitized = [
            self._strip_embedding_text(chunk) if isinstance(chunk, dict) else chunk
            for chunk in chunks
        ]
        payload = AddRecordsInput(chunks=sanitized, embeddings=embeddings)

        with self._write_lock:
            start_id = self._next_id()
            rows = [
                VectorRecord(
                    id=row_id,
                    source=chunk.source,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    vector=embedding,
                ).model_dump()
                for row_id, (chunk, embedding) in enumerate(
                    zip(payload.chunks, payload.embeddings), start=start_id
                )
            ]

            try:
                if self.table is None:
                    self.table = self.db.create_table(self.table_name, data=rows)
                else:
                    self.table.add(rows)
            except Exception as exc:
                print(f"Vector store write failed ({exc}); recreating table.")
                self._drop_table_unlocked()
                self.table = self.db.create_table(self.table_name, data=rows)

            self._update_fts_index()
            return len(rows)

    def _has_fts_index(self) -> bool:
        if self._fts_index_ready is not None:
            return self._fts_index_ready
        if self.table is None:
            return False
        for index in self.table.list_indices():
            columns = list(getattr(index, "columns", None) or [])
            name = str(getattr(index, "name", "")).lower()
            if "content" in columns or "content" in name or "fts" in name:
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
        self.table.create_index("content", config=FTS(), replace=True)
        self._fts_index_ready = True

    def _rows_to_documents(self, rows: list[dict]) -> list[dict]:
        return [
            RetrievedDocument.model_validate(row).model_dump(by_alias=True)
            for row in rows
        ]

    @staticmethod
    def merge_documents(*doc_lists: list[dict]) -> list[dict]:
        merged: list[dict] = []
        seen: set[int] = set()
        for docs in doc_lists:
            for doc in docs:
                doc_id = doc.get("id")
                if doc_id is not None:
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                merged.append(doc)
        return merged

    def sequential_search(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        if self.table is None:
            raise RuntimeError("No table yet. Call add() before sequential_search().")
        payload = VectorSearchInput(query_vector=query_vector, top_k=top_k)
        rows = (
            self.table.search(payload.query_vector)
            .metric("cosine")
            .limit(payload.top_k)
            .to_list()
        )
        return self._rows_to_documents(rows)

    def bm25(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if self.table is None:
            raise RuntimeError("No table yet. Call add() before bm25().")
        payload = KeywordSearchInput(query=query, top_k=top_k)
        if not self._has_fts_index():
            self._create_fts_index()
        rows = (
            self.table.search(payload.query, query_type="fts", fts_columns="content")
            .limit(payload.top_k)
            .to_list()
        )
        return self._rows_to_documents(rows)

    def hyde(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        payload = KeywordSearchInput(query=query, top_k=top_k)
        hypothetical_doc = generate_hypothetical_document(payload.query)
        query_vector = self.embedder.embed_query(hypothetical_doc)
        return self.sequential_search(query_vector, top_k=payload.top_k)

    async def ahybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        payload = HybridSearchInput(query=query, top_k=top_k)
        query_vector = await asyncio.to_thread(
            self.embedder.embed_query, payload.query
        )
        dense_docs, sparse_docs = await asyncio.gather(
            asyncio.to_thread(
                self.sequential_search, query_vector, payload.top_k
            ),
            asyncio.to_thread(self.bm25, payload.query, payload.top_k), 
        )
        return self.merge_documents(dense_docs, sparse_docs) 

    async def ahyde(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        payload = KeywordSearchInput(query=query, top_k=top_k)
        return await asyncio.to_thread(self.hyde, payload.query, payload.top_k)


def get_vector_store(
    db_path: str | Path = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    *,
    embedder=None,
) -> VectorStore:
    global _store
    resolved = Path(db_path)
    if (
        _store is None
        or _store.db_path != resolved
        or _store.table_name != table_name
    ):
        _store = VectorStore(
            db_path=resolved,
            table_name=table_name,
            embedder=embedder,
        )
    elif embedder is not None:
        _store.embedder = embedder
    return _store
