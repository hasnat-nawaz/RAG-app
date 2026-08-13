"""LanceDB vector store for RAG chunks."""

import json
from pathlib import Path

import lancedb
from lancedb.index import FTS

from embedding import embed_query


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "lanceDB"
DEFAULT_TABLE_NAME = "EmbeddingsTable"


class VectorStore:
    # Connect to (or create) a local LanceDB directory and remember the table name.
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

    # Open the table when it already exists; otherwise leave it unset until add().
    def _open_table_if_exists(self):
        if self.table_name in self.db.list_tables().tables:
            return self.db.open_table(self.table_name)
        return None

    # Next id is 1 on an empty table, otherwise one past the current row count.
    def _next_id(self) -> int:
        if self.table is None:
            return 1
        return self.table.count_rows() + 1

    # Turn one chunk + one embedding into a LanceDB row.
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

    # Append chunks + embeddings. Creates the table on first add, then appends after that.
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

        print(f"Adding {len(rows)} row(s) to LanceDB table '{self.table_name}' "
              f"(ids {start_id}-{start_id + len(rows) - 1})...")

        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=rows)
        else:
            self.table.add(rows)

        self._update_fts_index()

        total_rows = self.table.count_rows()
        print(f"Done. Total rows in table: {total_rows}")
        return total_rows

    # True when a full-text (BM25) index already exists on content.
    def _has_fts_index(self) -> bool:
        if self.table is None:
            return False
        for index in self.table.list_indices():
            columns = list(getattr(index, "columns", None) or [])
            name = str(getattr(index, "name", "")).lower()
            if "content" in columns or "content" in name or "fts" in name:
                return True
        return False

    # Create the FTS index once; later adds fold new rows in with optimize().
    def _update_fts_index(self) -> None:
        if self._has_fts_index():
            print("Updating FTS index with new rows...")
            self.table.optimize()
            print("FTS index updated.")
            return
        self._create_fts_index()

    # Build a BM25 full-text index on the content column.
    def _create_fts_index(self) -> None:
        print("Building FTS (BM25) index on 'content'...")
        self.table.create_index("content", config=FTS(), replace=True)
        print("FTS index ready.")

    # Exact KNN over all rows (no ANN index). Cosine distance, closest `top_k` vectors.
    def sequential_search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
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

    # BM25 keyword search over chunk content.
    def bm25(self, query: str, top_k: int = 5) -> list[dict]:
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


# Local test: load swarm chunks + embeddings and add them to the database.
if __name__ == "__main__":
    storage_dir = Path(__file__).resolve().parents[1] / "storage" / "output_texts"
    chunks_path = storage_dir / "swarm.chunks.json"
    embeddings_path = storage_dir / "swarm.embeddings.json"

    try:
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = json.loads(embeddings_path.read_text(encoding="utf-8"))

        store = VectorStore()
        total_rows = store.add(chunks, embeddings)
        print(f"Chunks: {len(chunks)}")
        print(f"Embeddings: {len(embeddings)}")
        print(f"Total rows: {total_rows}")
        print(f"DB path: {store.db_path}")

        query = "how many drones minimum should be used for the tasks"

        print("\n--- BM25 ---")
        bm25_hits = store.bm25(query, top_k=5)
        for hit in bm25_hits:
            hit.pop("vector", None)
            print(hit)

        print("\n--- Sequential search ---")
        query_vector = embed_query(query)
        knn_hits = store.sequential_search(query_vector, top_k=5)
        for hit in knn_hits:
            hit.pop("vector", None)
            print(hit)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
