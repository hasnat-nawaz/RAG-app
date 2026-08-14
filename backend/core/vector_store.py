"""LanceDB vector store for RAG chunks."""

import bootstrap  # noqa: F401

from pathlib import Path

import lancedb
from lancedb.index import FTS

from embedding import embed_query
from query_optimization.hypothetical_document import generate_hypothetical_document


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "lanceDB"
DEFAULT_TABLE_NAME = "EmbeddingsTable"
DEFAULT_TOP_K = 10

_store: "VectorStore | None" = None


def get_vector_store(
    db_path: str | Path = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
) -> "VectorStore":
    # Reuse one LanceDB connection + open table for the whole process.
    global _store
    if _store is None:
        _store = VectorStore(db_path=db_path, table_name=table_name)
    return _store


class VectorStore:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)
        self.table = self._open_table_if_exists()
        self._fts_index_ready: bool | None = None

    def _open_table_if_exists(self):
        if self.table_name in self.db.list_tables().tables:
            return self.db.open_table(self.table_name)
        return None

    def reset_table(self) -> None:
        if self.table_name in self.db.list_tables().tables:
            self.db.drop_table(self.table_name)
        self.table = None
        self._fts_index_ready = None

    def _next_id(self) -> int:
        if self.table is None:
            return 1
        return self.table.count_rows() + 1

    def _build_row(self, chunk: dict, embedding: list[float], row_id: int) -> dict:
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk {row_id} is not a dict.")
        if "content" not in chunk or not isinstance(chunk["content"], str):
            raise ValueError(f"Chunk {row_id} is missing string 'content'.")
        if "source" not in chunk or not isinstance(chunk["source"], str) or not chunk["source"].strip():
            raise ValueError(f"Chunk {row_id} is missing string 'source'.")
        if "metadata" not in chunk:
            raise ValueError(f"Chunk {row_id} is missing 'metadata'.")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Embedding {row_id} is missing or empty.")

        return {
            "id": row_id,
            "source": chunk["source"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "vector": embedding,
        }

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if not chunks:
            raise ValueError("add expects a non-empty list of chunk dicts.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        start_id = self._next_id()
        rows = [
            self._build_row(chunk, embedding, row_id=row_id)
            for row_id, (chunk, embedding) in enumerate(
                zip(chunks, embeddings), start=start_id
            )
        ]

        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=rows)
        else:
            self.table.add(rows)

        self._update_fts_index()
        return self.table.count_rows()

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

    def sequential_search(self, query_vector: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if self.table is None:
            raise RuntimeError("No table yet. Call add() before sequential_search().")
        if not query_vector:
            raise ValueError("sequential_search expects a non-empty query vector.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        return (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

    def bm25(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if self.table is None:
            raise RuntimeError("No table yet. Call add() before bm25().")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("bm25 expects a non-empty query string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not self._has_fts_index():
            self._create_fts_index()

        return (
            self.table.search(query, query_type="fts", fts_columns="content")
            .limit(top_k)
            .to_list()
        )

    def hyde(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("hyde expects a non-empty query string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        hypothetical_doc = generate_hypothetical_document(query)
        query_vector = embed_query(hypothetical_doc)
        return self.sequential_search(query_vector, top_k=top_k)
