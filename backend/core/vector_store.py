"""LanceDB vector store for RAG chunks."""

import bootstrap  # noqa: F401

from pathlib import Path

import lancedb
from lancedb.index import FTS

from embedding import embed_query
from query_optimization.hypothetical_document import generate_hypothetical_document
from query_optimization.query_expansion import expand_query
from query_optimization.query_rewriting import rewrite_query
from reranker import rerank


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "lanceDB"
DEFAULT_TABLE_NAME = "EmbeddingsTable"
DEFAULT_TOP_K = 10


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

    # BM25 keyword search over chunk content.
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

    # HyDE: generate a hypothetical doc, embed it, then run semantic search.
    def hyde(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("hyde expects a non-empty query string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        hypothetical_doc = generate_hypothetical_document(query)
        query_vector = embed_query(hypothetical_doc)
        return self.sequential_search(query_vector, top_k=top_k)


# Local test: run all retrieval paths and save results to storage/outputs.txt.
if __name__ == "__main__":
    import time

    storage_dir = Path(__file__).resolve().parents[1] / "storage"
    output_path = storage_dir / "outputs.txt"
    top_k = DEFAULT_TOP_K

    # Edit this query to test retrieval.
    query = "What's the exact procedure for the semi-autonomous fleet control task? Like, how does the swarm move, what maneuvers can they actually perform in manoeuvre mode versus herd mode, and what are the specific rules and instructions the referees or pilot give using the controller during this?"

    def _format_hits(hits: list[dict]) -> str:
        lines = []
        for i, hit in enumerate(hits, start=1):
            hit = {k: v for k, v in hit.items() if k != "vector"}
            lines.append(f"  [{i}] id={hit.get('id')} source={hit.get('source')}")
            lines.append(f"      metadata={hit.get('metadata')}")
            lines.append(f"      content={hit.get('content', '')[:300]}...")
            if "_distance" in hit:
                lines.append(f"      distance={hit['_distance']}")
            if "_score" in hit:
                lines.append(f"      score={hit['_score']}")
            if "rerank_score" in hit:
                lines.append(f"      rerank_score={hit['rerank_score']}")
        return "\n".join(lines) if lines else "  (no results)"

    try:
        store = VectorStore()
        if store.table is None:
            raise RuntimeError("No table found. Populate the database with add() first.")

        sections: list[str] = [f"QUERY: {query}\n"]

        # 1) Rewrite → sequential search
        print("\n=== REWRITE → SEQUENTIAL ===")
        t_path = time.perf_counter()
        t0 = time.perf_counter()
        rewritten = rewrite_query(query)
        print(f"  rewrite_query: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        rewrite_vector = embed_query(rewritten)
        print(f"  embed_query: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        rewrite_hits = store.sequential_search(rewrite_vector, top_k=top_k)
        print(f"  sequential_search: {time.perf_counter() - t0:.3f}s")
        print(f"  total: {time.perf_counter() - t_path:.3f}s")
        sections.append("=== REWRITE → SEQUENTIAL ===")
        sections.append(f"Optimized query:\n{rewritten}\n")
        sections.append(f"Results:\n{_format_hits(rewrite_hits)}\n")

        # 2) Expansion → BM25
        print("\n=== EXPANSION → BM25 ===")
        t_path = time.perf_counter()
        t0 = time.perf_counter()
        expanded = expand_query(query)
        print(f"  expand_query: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        bm25_hits = store.bm25(expanded, top_k=top_k)
        print(f"  bm25: {time.perf_counter() - t0:.3f}s")
        print(f"  total: {time.perf_counter() - t_path:.3f}s")
        sections.append("=== EXPANSION → BM25 ===")
        sections.append(f"Optimized query:\n{expanded}\n")
        sections.append(f"Results:\n{_format_hits(bm25_hits)}\n")

        # 3) HyDE → sequential search
        print("\n=== HYDE → SEQUENTIAL ===")
        t_path = time.perf_counter()
        t0 = time.perf_counter()
        hypothetical_doc = generate_hypothetical_document(query)
        print(f"  generate_hypothetical_document: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        hyde_vector = embed_query(hypothetical_doc)
        print(f"  embed_query: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()
        hyde_hits = store.sequential_search(hyde_vector, top_k=top_k)
        print(f"  sequential_search: {time.perf_counter() - t0:.3f}s")
        print(f"  total: {time.perf_counter() - t_path:.3f}s")
        sections.append("=== HYDE → SEQUENTIAL ===")
        sections.append(f"Hypothetical document:\n{hypothetical_doc}\n")
        sections.append(f"Results:\n{_format_hits(hyde_hits)}\n")

        # 4) Merge all paths → rerank (10–30 docs depending on overlap)
        print("\n=== MERGE → RERANK ===")
        t_path = time.perf_counter()
        merged_hits = rewrite_hits + bm25_hits + hyde_hits
        print(f"  merged: {len(merged_hits)} doc(s) from 3 retrievers")
        t0 = time.perf_counter()
        reranked_hits = rerank(query, merged_hits)
        print(f"  rerank: {time.perf_counter() - t0:.3f}s")
        print(f"  top {len(reranked_hits)}: {len(reranked_hits)}/{len(merged_hits)} doc(s)")
        print(f"  total: {time.perf_counter() - t_path:.3f}s")
        sections.append("=== MERGE → RERANK ===")
        sections.append(f"Merged: {len(merged_hits)} | Top reranked: {len(reranked_hits)}\n")
        sections.append(f"Results:\n{_format_hits(reranked_hits)}\n")

        output_path.write_text("\n".join(sections), encoding="utf-8")
        print(f"Saved results to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
