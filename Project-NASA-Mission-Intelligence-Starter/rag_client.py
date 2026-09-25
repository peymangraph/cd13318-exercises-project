"""ChromaDB retrieval utilities for the NASA Mission Intelligence project."""

import math
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


def _build_term_weights(query: str, documents: List[str]) -> Dict[str, float]:
    """Build lightweight IDF-style weights for the query terms across the active corpus."""
    query_terms = set(_keyword_terms(query))
    if not query_terms:
        return {}

    document_count = max(len(documents), 1)
    document_frequency = {term: 0 for term in query_terms}

    for document in documents:
        document_terms = set(_keyword_terms(document))
        for term in query_terms & document_terms:
            document_frequency[term] += 1

    return {
        term: 1.0 + math.log((document_count + 1) / (document_frequency[term] + 1))
        for term in query_terms
    }


def _keyword_overlap_score(
    query: str,
    document: str,
    term_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score weighted keyword and phrase overlap between a query and document."""
    query_terms = _keyword_terms(query)
    if not query_terms:
        return 0.0

    document_terms = _keyword_terms(document)
    if not document_terms:
        return 0.0

    query_set = set(query_terms)
    document_set = set(document_terms)
    weights = term_weights or {term: 1.0 for term in query_set}

    total_weight = sum(weights.get(term, 1.0) for term in query_set)
    matched_weight = sum(
        weights.get(term, 1.0)
        for term in query_set
        if term in document_set
    )
    unigram_score = matched_weight / total_weight if total_weight else 0.0

    query_bigrams = set(zip(query_terms, query_terms[1:]))
    document_bigrams = set(zip(document_terms, document_terms[1:]))
    bigram_score = (
        len(query_bigrams & document_bigrams) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )

    return (0.85 * unigram_score) + (0.15 * bigram_score)


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    """Return cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    return dot / denominator if denominator else 0.0


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
    """Merge semantic and full-corpus keyword candidates, then hybrid-rerank them."""
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

    clean_query = query.strip()
    query_embedding = _embed_query(clean_query, api_key, embedding_model)

    try:
        collection_size = collection.count()
    except Exception:
        collection_size = n_results

    if collection_size == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    semantic_candidate_count = min(collection_size, max(n_results * 10, 50))
    semantic_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": semantic_candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        semantic_kwargs["where"] = where

    semantic_raw = collection.query(**semantic_kwargs)
    semantic_ids = (semantic_raw.get("ids") or [[]])[0]
    semantic_documents = (semantic_raw.get("documents") or [[]])[0]
    semantic_metadatas = (semantic_raw.get("metadatas") or [[]])[0]
    semantic_distances = (semantic_raw.get("distances") or [[]])[0]

    corpus_kwargs = {"include": ["documents", "metadatas"]}
    if where is not None:
        corpus_kwargs["where"] = where
    corpus = collection.get(**corpus_kwargs)

    corpus_ids = list(corpus.get("ids") or [])
    corpus_documents = list(corpus.get("documents") or [])
    corpus_metadatas = list(corpus.get("metadatas") or [])

    term_weights = _build_term_weights(clean_query, corpus_documents)
    lexical_candidate_count = max(n_results * 10, 50)

    lexical_ranked = []
    for idx, document in enumerate(corpus_documents):
        if not document:
            continue
        score = _keyword_overlap_score(clean_query, document, term_weights)
        if score <= 0:
            continue

        doc_id = corpus_ids[idx] if idx < len(corpus_ids) else f"lexical-{idx}"
        metadata = (
            corpus_metadatas[idx]
            if idx < len(corpus_metadatas) and corpus_metadatas[idx]
            else {}
        )
        lexical_ranked.append((score, doc_id, document, metadata))

    lexical_ranked.sort(key=lambda item: item[0], reverse=True)
    lexical_ranked = lexical_ranked[:lexical_candidate_count]

    candidates: Dict[str, Dict[str, Any]] = {}

    for idx, document in enumerate(semantic_documents):
        if not document:
            continue
        doc_id = semantic_ids[idx] if idx < len(semantic_ids) else f"semantic-{idx}"
        metadata = (
            semantic_metadatas[idx]
            if idx < len(semantic_metadatas) and semantic_metadatas[idx]
            else {}
        )
        distance = semantic_distances[idx] if idx < len(semantic_distances) else None
        candidates[doc_id] = {
            "id": doc_id,
            "document": document,
            "metadata": metadata,
            "distance": distance,
        }

    for keyword_score, doc_id, document, metadata in lexical_ranked:
        candidate = candidates.setdefault(
            doc_id,
            {
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "distance": None,
            },
        )
        candidate["keyword_score"] = keyword_score

    candidate_ids = list(candidates)
    embedding_lookup: Dict[str, List[float]] = {}
    if candidate_ids:
        embedding_rows = collection.get(ids=candidate_ids, include=["embeddings"])
        embedding_ids = list(embedding_rows.get("ids") or [])
        embedding_values = embedding_rows.get("embeddings")
        if embedding_values is not None:
            for doc_id, embedding in zip(embedding_ids, embedding_values):
                if embedding is not None:
                    embedding_lookup[doc_id] = list(embedding)

    cosine_scores = []
    for candidate in candidates.values():
        keyword_score = candidate.get("keyword_score")
        if keyword_score is None:
            keyword_score = _keyword_overlap_score(
                clean_query,
                candidate["document"],
                term_weights,
            )
        candidate["keyword_score"] = keyword_score

        embedding = embedding_lookup.get(candidate["id"])
        cosine_score = _cosine_similarity(query_embedding, embedding) if embedding else 0.0
        candidate["cosine_score"] = cosine_score
        cosine_scores.append(cosine_score)

    min_cosine = min(cosine_scores) if cosine_scores else 0.0
    max_cosine = max(cosine_scores) if cosine_scores else 0.0
    cosine_span = max_cosine - min_cosine

    ranked = []
    for candidate in candidates.values():
        semantic_score = (
            1.0
            if cosine_span == 0 and candidate["cosine_score"] != 0.0
            else (
                (candidate["cosine_score"] - min_cosine) / cosine_span
                if cosine_span
                else 0.0
            )
        )
        keyword_score = candidate["keyword_score"]

        hybrid_score = (0.65 * semantic_score) + (0.35 * keyword_score)
        distance = candidate["distance"]
        if not isinstance(distance, (int, float)):
            distance = 1.0 - candidate["cosine_score"]

        ranked.append(
            (
                hybrid_score,
                keyword_score,
                semantic_score,
                distance,
                candidate["id"],
                candidate["document"],
                candidate["metadata"],
            )
        )

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))

    unique = []
    selected_documents: List[str] = []
    seen_exact = set()

    for item in ranked:
        document = item[5]
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
        "ids": [[item[4] for item in unique]],
        "documents": [[item[5] for item in unique]],
        "metadatas": [[item[6] for item in unique]],
        "distances": [[item[3] for item in unique]],
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
