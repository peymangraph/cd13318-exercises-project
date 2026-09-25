"""ChromaDB retrieval utilities for the NASA Mission Intelligence project."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI


def discover_chroma_backends() -> Dict[str, Dict[str, Any]]:
    """Discover persistent ChromaDB directories and collections near the app."""
    backends: Dict[str, Dict[str, Any]] = {}
    current_dir = Path(".")

    candidates = {
        p.resolve()
        for p in current_dir.iterdir()
        if p.is_dir() and ("chroma" in p.name.lower() or p.name.endswith("_db"))
    }

    for directory in sorted(candidates, key=lambda p: str(p)):
        try:
            client = chromadb.PersistentClient(path=str(directory))
            collections = client.list_collections()
            for collection_info in collections:
                name = getattr(collection_info, "name", str(collection_info))
                collection = client.get_collection(name=name)
                key = f"{directory.name}:{name}"
                backends[key] = {
                    "directory": str(directory),
                    "collection_name": name,
                    "display_name": f"{name} ({directory.name}, {collection.count()} chunks)",
                    "document_count": collection.count(),
                }
        except Exception as exc:
            key = f"{directory.name}:error"
            backends[key] = {
                "directory": str(directory),
                "collection_name": "",
                "display_name": f"{directory.name} (unavailable: {str(exc)[:80]})",
                "document_count": 0,
                "error": str(exc),
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Connect to a persistent Chroma collection.

    Returns (collection, success, error_message) for direct use by Streamlit.
    """
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as exc:
        return None, False, str(exc)


def _embed_query(query: str, openai_key: str, embedding_model: str) -> List[float]:
    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    client = OpenAI(api_key=openai_key, base_url=base_url)
    response = client.embeddings.create(model=embedding_model, input=query)
    return response.data[0].embedding


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


def _keyword_terms(text: str) -> List[str]:
    """Extract meaningful lowercase query/document terms for lexical scoring."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _keyword_overlap_score(query: str, document: str) -> float:
    """Score how strongly a document overlaps the query's important terms and phrases."""
    query_terms = _keyword_terms(query)
    if not query_terms:
        return 0.0

    document_terms = _keyword_terms(document)
    if not document_terms:
        return 0.0

    query_set = set(query_terms)
    document_set = set(document_terms)
    unigram_score = len(query_set & document_set) / len(query_set)

    query_bigrams = set(zip(query_terms, query_terms[1:]))
    document_bigrams = set(zip(document_terms, document_terms[1:]))
    bigram_score = (
        len(query_bigrams & document_bigrams) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )

    return (0.8 * unigram_score) + (0.2 * bigram_score)


def _normalized_tokens(text: str) -> set[str]:
    """Normalize a chunk into tokens for exact and near-duplicate detection."""
    return set(" ".join(text.lower().split()).split())


def _is_near_duplicate(candidate: str, selected_documents: List[str], threshold: float = 0.9) -> bool:
    """Return True when a candidate substantially duplicates an already selected chunk."""
    candidate_tokens = _normalized_tokens(candidate)
    if not candidate_tokens:
        return True

    for selected in selected_documents:
        selected_tokens = _normalized_tokens(selected)
        if not selected_tokens:
            continue
        union = candidate_tokens | selected_tokens
        if not union:
            continue
        similarity = len(candidate_tokens & selected_tokens) / len(union)
        if similarity >= threshold:
            return True
    return False


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    openai_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
) -> Optional[Dict[str, Any]]:
    """Retrieve and rerank chunks using semantic similarity plus keyword overlap."""
    if collection is None:
        raise ValueError("A ChromaDB collection is required.")
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    api_key = openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required to embed the user query.")

    where = None
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        where = {"mission": mission_filter}

    query_embedding = _embed_query(query.strip(), api_key, embedding_model)

    try:
        collection_size = collection.count()
    except Exception:
        collection_size = n_results

    if collection_size == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Retrieve a substantially larger pool than the final top-k so that
    # deduplication does not leave the final context dominated by overlapping chunks.
    candidate_count = min(collection_size, max(n_results * 10, 50))
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    finite_distances = [
        float(distance)
        for distance in distances
        if isinstance(distance, (int, float)) and distance != float("inf")
    ]
    min_distance = min(finite_distances) if finite_distances else 0.0
    max_distance = max(finite_distances) if finite_distances else 0.0
    distance_span = max_distance - min_distance

    ranked = []
    for idx, document in enumerate(documents):
        if not document:
            continue

        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        distance = distances[idx] if idx < len(distances) else float("inf")
        doc_id = ids[idx] if idx < len(ids) else f"result-{idx}"

        if isinstance(distance, (int, float)) and distance != float("inf"):
            semantic_score = (
                1.0
                if distance_span == 0
                else 1.0 - ((float(distance) - min_distance) / distance_span)
            )
        else:
            semantic_score = 0.0

        keyword_score = _keyword_overlap_score(query, document)

        # Semantic similarity remains the primary signal while lexical overlap
        # promotes chunks that explicitly contain important query terms/phrases.
        hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score)
        ranked.append((hybrid_score, distance, doc_id, document, metadata))

    ranked.sort(key=lambda item: (-item[0], item[1]))

    # Walk the hybrid ranking and keep only distinct chunks so the final context
    # contains the strongest diverse evidence instead of overlapping windows.
    unique = []
    selected_documents: List[str] = []
    seen_exact = set()

    for item in ranked:
        document = item[3]
        normalized = " ".join(document.lower().split())
        if normalized in seen_exact:
            continue
        if _is_near_duplicate(document, selected_documents):
            continue

        seen_exact.add(normalized)
        selected_documents.append(document)
        unique.append(item)

        if len(unique) >= n_results:
            break

    return {
        "ids": [[item[2] for item in unique]],
        "documents": [[item[3] for item in unique]],
        "metadatas": [[item[4] for item in unique]],
        "distances": [[item[1] for item in unique]],
    }


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Build one source-attributed context string for the LLM."""
    if not documents:
        return ""

    context_parts = ["NASA RETRIEVED CONTEXT"]
    seen = set()

    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        if not document:
            continue
        normalized = " ".join(document.lower().split())
        if normalized[:500] in seen:
            continue
        seen.add(normalized[:500])

        metadata = metadata or {}
        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source") or metadata.get("file_path") or "unknown source"
        file_path = metadata.get("file_path", "")
        category = metadata.get("document_category", "document").replace("_", " ").title()
        chunk_index = metadata.get("chunk_index")

        attribution = f"[Source {index}] Mission: {mission} | Source: {source} | Category: {category}"
        if chunk_index is not None:
            attribution += f" | Chunk: {chunk_index}"
        if file_path:
            attribution += f" | File: {file_path}"

        context_parts.append("\n" + "=" * 72)
        context_parts.append(attribution)
        context_parts.append("-" * 72)
        context_parts.append(document.strip())

    return "\n".join(context_parts).strip()
