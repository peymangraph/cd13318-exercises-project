"""ChromaDB retrieval utilities for the NASA Mission Intelligence project."""

import math
import os
import re
from collections import Counter
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



def _reformulate_retrieval_query(
    query: str,
    mission_filter: Optional[str],
    openai_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Create a compact retrieval-oriented query while preserving user intent.

    This is a retrieval-only transformation. The original user question is still
    passed unchanged to the answer-generation model. If reformulation fails, the
    original query is returned so retrieval continues normally.
    """
    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    client = OpenAI(api_key=openai_key, base_url=base_url)

    mission_context = ""
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        mission_context = f"Mission filter: {mission_filter}. "

    system_prompt = (
        "You rewrite user questions into concise search queries for retrieval-augmented "
        "generation. Preserve named entities, mission names, dates, technical terms, and "
        "the user's intent. Add useful synonyms or closely related technical vocabulary "
        "only when they are strongly implied by the question. Do not answer the question. "
        "Do not invent facts. Return one compact retrieval query only."
    )

    user_prompt = (
        f"{mission_context}"
        f"Original question: {query}\n"
        "Rewrite this as a retrieval query of roughly 8 to 24 informative words or short phrases."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            return query
        # Keep the original wording alongside the rewrite so precise terms are never lost.
        return f"{query}\n{rewritten}"
    except Exception:
        return query


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "during", "happened",
}


def _keyword_terms(text: str) -> List[str]:
    """Extract meaningful lowercase query/document terms for lexical scoring."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _lexical_query(query: str, mission_filter: Optional[str]) -> str:
    """Remove mission-filter terms that add no lexical discrimination."""
    if not mission_filter or mission_filter.lower() in {"all", "any", "none"}:
        return query

    mission_terms = set(_keyword_terms(mission_filter.replace("_", " ")))
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    filtered = [token for token in tokens if token not in mission_terms]
    return " ".join(filtered) or query


def _bm25_scores(
    query: str,
    documents: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Return BM25-style lexical relevance scores for every document."""
    query_terms = list(dict.fromkeys(_keyword_terms(query)))
    if not query_terms or not documents:
        return [0.0] * len(documents)

    tokenized_documents = [_keyword_terms(document or "") for document in documents]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = (
        sum(document_lengths) / len(document_lengths)
        if document_lengths
        else 0.0
    )
    if average_length <= 0:
        return [0.0] * len(documents)

    document_frequency = {term: 0 for term in query_terms}
    term_frequencies = []

    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        for term in query_terms:
            if frequencies.get(term, 0) > 0:
                document_frequency[term] += 1

    document_count = len(documents)
    idf = {
        term: math.log(
            1.0
            + (
                (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
        )
        for term in query_terms
    }

    scores: List[float] = []
    for frequencies, document_length in zip(term_frequencies, document_lengths):
        score = 0.0
        length_normalization = k1 * (
            1.0 - b + b * (document_length / average_length)
        )

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            score += idf[term] * (
                (term_frequency * (k1 + 1.0))
                / (term_frequency + length_normalization)
            )

        scores.append(score)

    return scores


def _rrf_score(
    semantic_rank: Optional[int],
    lexical_rank: Optional[int],
    rrf_k: int = 60,
) -> float:
    """Fuse semantic and lexical ranks using reciprocal-rank fusion."""
    score = 0.0
    if semantic_rank is not None:
        score += 1.0 / (rrf_k + semantic_rank)
    if lexical_rank is not None:
        score += 1.0 / (rrf_k + lexical_rank)
    return score


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
    """Build a balanced semantic/BM25 pool, expand neighbors, and keep top-k."""
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
    retrieval_query = _reformulate_retrieval_query(
        clean_query,
        mission_filter,
        api_key,
    )
    lexical_query = _lexical_query(retrieval_query, mission_filter)
    query_embedding = _embed_query(retrieval_query, api_key, embedding_model)

    try:
        collection_size = collection.count()
    except Exception:
        collection_size = n_results

    if collection_size == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Pull a broad semantic ranking.
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

    # Load every chunk in the selected mission/corpus for BM25 and neighbor lookup.
    corpus_kwargs = {"include": ["documents", "metadatas"]}
    if where is not None:
        corpus_kwargs["where"] = where
    corpus = collection.get(**corpus_kwargs)

    corpus_ids = list(corpus.get("ids") or [])
    corpus_documents = list(corpus.get("documents") or [])
    corpus_metadatas = list(corpus.get("metadatas") or [])

    corpus_by_id: Dict[str, Dict[str, Any]] = {}
    neighbor_lookup: Dict[tuple, Dict[str, Any]] = {}

    for idx, document in enumerate(corpus_documents):
        if not document:
            continue
        doc_id = corpus_ids[idx] if idx < len(corpus_ids) else f"corpus-{idx}"
        metadata = (
            corpus_metadatas[idx]
            if idx < len(corpus_metadatas) and corpus_metadatas[idx]
            else {}
        )
        row = {
            "id": doc_id,
            "document": document,
            "metadata": metadata,
            "distance": None,
            "semantic_rank": None,
            "lexical_rank": None,
            "bm25_score": 0.0,
            "neighbor_of": None,
        }
        corpus_by_id[doc_id] = row

        file_path = metadata.get("file_path")
        chunk_index = metadata.get("chunk_index")
        if file_path is not None and isinstance(chunk_index, int):
            neighbor_lookup[(str(file_path), chunk_index)] = row

    # Rank the complete filtered corpus lexically with BM25.
    bm25_scores = _bm25_scores(lexical_query, corpus_documents)
    lexical_ranked = [
        (score, idx)
        for idx, score in enumerate(bm25_scores)
        if score > 0
    ]
    lexical_ranked.sort(key=lambda item: item[0], reverse=True)

    # Use equal-depth semantic and lexical seed pools so one retrieval method
    # cannot crowd the other out before final selection.
    per_source_seed_count = max(n_results * 2, 20)
    semantic_seed_count = min(len(semantic_documents), per_source_seed_count)
    lexical_seed_count = min(len(lexical_ranked), per_source_seed_count)

    candidates: Dict[str, Dict[str, Any]] = {}

    for semantic_rank in range(1, semantic_seed_count + 1):
        idx = semantic_rank - 1
        document = semantic_documents[idx]
        if not document:
            continue

        doc_id = (
            semantic_ids[idx]
            if idx < len(semantic_ids)
            else f"semantic-{idx}"
        )
        metadata = (
            semantic_metadatas[idx]
            if idx < len(semantic_metadatas) and semantic_metadatas[idx]
            else {}
        )
        distance = (
            semantic_distances[idx]
            if idx < len(semantic_distances)
            else None
        )

        candidate = dict(
            corpus_by_id.get(
                doc_id,
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "semantic_rank": None,
                    "lexical_rank": None,
                    "bm25_score": 0.0,
                    "neighbor_of": None,
                },
            )
        )
        candidate["document"] = document
        candidate["metadata"] = metadata
        candidate["distance"] = distance
        candidate["semantic_rank"] = semantic_rank
        candidates[doc_id] = candidate

    for lexical_rank in range(1, lexical_seed_count + 1):
        bm25_score, idx = lexical_ranked[lexical_rank - 1]
        if idx >= len(corpus_documents) or not corpus_documents[idx]:
            continue

        doc_id = (
            corpus_ids[idx]
            if idx < len(corpus_ids)
            else f"lexical-{idx}"
        )
        candidate = candidates.get(doc_id)
        if candidate is None:
            candidate = dict(corpus_by_id[doc_id])

        candidate["lexical_rank"] = lexical_rank
        candidate["bm25_score"] = bm25_score
        candidates[doc_id] = candidate

    seed_ids = list(candidates)

    # Add the immediately adjacent chunks around every seed so the LLM receives
    # enough local transcript/document continuity to understand the event.
    for seed_id in seed_ids:
        seed = candidates[seed_id]
        metadata = seed.get("metadata") or {}
        file_path = metadata.get("file_path")
        chunk_index = metadata.get("chunk_index")
        if file_path is None or not isinstance(chunk_index, int):
            continue

        for delta in (-1, 1):
            neighbor = neighbor_lookup.get((str(file_path), chunk_index + delta))
            if not neighbor:
                continue

            neighbor_id = neighbor["id"]
            if neighbor_id in candidates:
                continue

            expanded = dict(neighbor)
            expanded["neighbor_of"] = seed_id
            expanded["semantic_rank"] = seed.get("semantic_rank")
            expanded["lexical_rank"] = seed.get("lexical_rank")
            expanded["bm25_score"] = seed.get("bm25_score", 0.0)
            candidates[neighbor_id] = expanded

    # Give semantic and lexical rank equal standing. A chunk that is strong in
    # either channel remains competitive; appearing in both provides a small bonus.
    ranked = []
    for candidate in candidates.values():
        semantic_rank = candidate.get("semantic_rank")
        lexical_rank = candidate.get("lexical_rank")

        semantic_strength = (
            1.0 / semantic_rank if isinstance(semantic_rank, int) and semantic_rank > 0 else 0.0
        )
        lexical_strength = (
            1.0 / lexical_rank if isinstance(lexical_rank, int) and lexical_rank > 0 else 0.0
        )

        primary_strength = max(semantic_strength, lexical_strength)
        secondary_strength = min(semantic_strength, lexical_strength)
        balanced_score = primary_strength + (0.15 * secondary_strength)

        if candidate.get("neighbor_of"):
            balanced_score *= 0.92

        ranked.append(
            (
                balanced_score,
                candidate.get("bm25_score", 0.0),
                semantic_rank or float("inf"),
                lexical_rank or float("inf"),
                candidate["id"],
                candidate["document"],
                candidate["metadata"],
                candidate["distance"],
            )
        )

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

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
        "distances": [[item[7] for item in unique]],
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
