"""RAG query pipeline entry point (retrieval → rerank → generate)."""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

BACKEND_DIR = Path(__file__).resolve().parent
CORE_DIR = BACKEND_DIR / "core"
sys.path.insert(0, str(CORE_DIR))

import bootstrap  # noqa: F401

from embedding import embed_query
from generation import generate_response
from query_optimization.hypothetical_document import generate_hypothetical_document
from query_optimization.query_expansion import expand_query
from query_optimization.query_rewriting import rewrite_query
from reranker import rerank, start_reranker_warmup, wait_for_reranker_warmup
from vector_store import get_vector_store, DEFAULT_TOP_K


# Edit this before running.
QUERY = (
   "Under the rules of the Dynamic Swarm Capability Task and General Security guidelines, what is the exact fail-safe duration and required vehicle mode triggered upon lost control connection? Additionally, what are the strict rules, landing radius constraints, and timing/rejoining conditions when executing an 'add or remove an individual from the swarm' command via QR code? Finally, how does scoring and task continuity function if the number of active UAVs drops from 3 to 2 during flight across different types of missions?"
)

STORAGE_DIR = BACKEND_DIR / "storage"
RETRIEVED_DIR = STORAGE_DIR / "retrieved_docs"
RESPONSES_DIR = STORAGE_DIR / "responses"


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _section(title: str) -> None:
    print(f"\n{title}")


def _step(message: str) -> None:
    print(f"  {message}")


def _strip_vector(doc: dict) -> dict:
    return {key: value for key, value in doc.items() if key != "vector"}


def retrieve_and_rerank(query: str, run_id: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    _section("RETRIEVING DOCUMENTS")
    section_start = time.perf_counter()
    store = get_vector_store()
    if store.table is None:
        raise RuntimeError("No database table found. Run indexing first.")

    details: dict = {"query": query, "paths": {}}

    start_reranker_warmup()

    step_start = time.perf_counter()
    _step("rewrite query")
    rewritten = rewrite_query(query)
    _step(f"time taken = {_format_seconds(time.perf_counter() - step_start)}")

    embed_start = time.perf_counter()
    rewrite_vector = embed_query(rewritten)
    _step(f"embed query (rewrite path) = {_format_seconds(time.perf_counter() - embed_start)}")

    search_start = time.perf_counter()
    _step("sequential search (rewrite path)")
    rewrite_hits = store.sequential_search(rewrite_vector, top_k=top_k)
    details["paths"]["rewrite_sequential"] = {
        "optimized_query": rewritten,
        "retrieved": len(rewrite_hits),
    }
    _step(f"docs retrieved = {len(rewrite_hits)}")
    _step(f"search time = {_format_seconds(time.perf_counter() - search_start)}")

    step_start = time.perf_counter()
    _step("expand query")
    expanded = expand_query(query)
    _step(f"time taken = {_format_seconds(time.perf_counter() - step_start)}")

    search_start = time.perf_counter()
    _step("bm25 search (expansion path)")
    bm25_hits = store.bm25(expanded, top_k=top_k)
    details["paths"]["expansion_bm25"] = {
        "optimized_query": expanded,
        "retrieved": len(bm25_hits),
    }
    _step(f"docs retrieved = {len(bm25_hits)}")
    _step(f"search time = {_format_seconds(time.perf_counter() - search_start)}")

    step_start = time.perf_counter()
    _step("generate hypothetical document")
    hypothetical_doc = generate_hypothetical_document(query)
    _step(f"hypothetical document length = {len(hypothetical_doc)} chars")
    _step(f"time taken = {_format_seconds(time.perf_counter() - step_start)}")

    embed_start = time.perf_counter()
    hyde_vector = embed_query(hypothetical_doc)
    _step(f"embed query (hyde path) = {_format_seconds(time.perf_counter() - embed_start)}")

    search_start = time.perf_counter()
    _step("sequential search (hyde path)")
    hyde_hits = store.sequential_search(hyde_vector, top_k=top_k)
    details["paths"]["hyde_sequential"] = {
        "hypothetical_document": hypothetical_doc,
        "retrieved": len(hyde_hits),
    }
    _step(f"docs retrieved = {len(hyde_hits)}")
    _step(f"search time = {_format_seconds(time.perf_counter() - search_start)}")

    merged_hits = rewrite_hits + bm25_hits + hyde_hits
    _step(f"merged candidates = {len(merged_hits)}")

    wait_for_reranker_warmup()

    step_start = time.perf_counter()
    _step("reranking documents")
    reranked_hits = rerank(query, merged_hits)
    details["merged"] = len(merged_hits)
    details["reranked"] = len(reranked_hits)
    details["documents"] = [_strip_vector(doc) for doc in reranked_hits]
    _step(f"reranked docs selected = {len(reranked_hits)}")
    _step(f"rerank inference time = {_format_seconds(time.perf_counter() - step_start)}")

    retrieved_path = save_retrieved_docs(details, run_id)
    _step(f"retrieved docs saved to = {retrieved_path.name}")
    _step(f"time taken = {_format_seconds(time.perf_counter() - section_start)}")
    return reranked_hits


def generate_and_save_response(query: str, documents: list[dict], run_id: str) -> Path:
    _section("GENERATING RESPONSE")
    section_start = time.perf_counter()
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    answer = generate_response(query, documents)
    response_path = RESPONSES_DIR / f"response_{run_id}.txt"
    response_path.write_text(answer, encoding="utf-8")

    _step(f"response length = {len(answer)} characters")
    _step(f"saved to = {response_path.name}")
    _step(f"time taken = {_format_seconds(time.perf_counter() - section_start)}")
    return response_path


def save_retrieved_docs(details: dict, run_id: str) -> Path:
    RETRIEVED_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_path = RETRIEVED_DIR / f"retrieved_{run_id}.json"
    retrieved_path.write_text(
        json.dumps(details, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return retrieved_path


def main() -> None:
    pipeline_start = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    reranked_docs = retrieve_and_rerank(QUERY, run_id)
    generate_and_save_response(QUERY, reranked_docs, run_id)

    _section("PIPELINE COMPLETE")
    _step(f"time taken = {_format_seconds(time.perf_counter() - pipeline_start)}")


if __name__ == "__main__":
    main()
