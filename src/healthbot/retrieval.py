from typing import Any

from langchain_core.documents import Document

from healthbot.state import RetrievalDecision

from .ingestion import build_vector_store


def retrieve_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    store = build_vector_store()
    return store.similarity_search_with_score(query, k=k)


def retrieval_stats(results: list[tuple[Document, float]]) -> dict[str, Any]:
    if not results:
        return {
            "doc_count": 0,
            "best_score": None,
            "worst_score": None,
            "avg_score": None,
        }

    scores = [score for _, score in results]
    return {
        "doc_count": len(results),
        "best_score": min(scores),
        "worst_score": max(scores),
        "avg_score": sum(scores) / len(scores),
    }


def retrieve_documents(query: str, k: int = 4) -> list[Document]:
    store = build_vector_store()
    return store.similarity_search(query, k=k)


def format_context(documents: list[Document]) -> tuple[str, list[str]]:
    chunks = []
    citations: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("source_name", "unknown")
        source_url = doc.metadata.get(
            "source_url", doc.metadata.get("filename", "unknown")
        )
        chunk_id = doc.metadata.get("chunk_id", f"chunk-{idx}")
        citations.append(source_url)
        chunks.append(
            f"[Chunk {idx} | ID: {chunk_id} | Source: {source_name} | URL: {source_url}]\n"
            f"{doc.page_content}"
        )

    unique_citations = list(dict.fromkeys(citations))
    return "\n\n".join(chunks), unique_citations


def decide_retrieval_path(results: list[tuple[Document, float]]) -> RetrievalDecision:
    stats = retrieval_stats(results)

    if stats["doc_count"] == 0:
        return RetrievalDecision(
            use_local_only=False,
            needs_web_fallback=True,
            reason="no_local_matches",
        )

    if stats["best_score"] is not None and stats["best_score"] > 1.2:
        return RetrievalDecision(
            use_local_only=False,
            needs_web_fallback=True,
            reason="weak_similarity_scores",
        )

    return RetrievalDecision(
        use_local_only=True,
        needs_web_fallback=False,
        reason="local_context_sufficient",
    )
