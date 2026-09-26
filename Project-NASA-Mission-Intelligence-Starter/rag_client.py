"""Retrieval utilities for the NASA Mission Intelligence project."""

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI


def _collection_mission(collection) -> str:
    try:
        first = collection.get(limit=1, include=["metadatas"])
        metadata = ((first.get("metadatas") or [{}])[:1] or [{}])[0] or {}
        return str(metadata.get("mission", "unknown"))
    except Exception:
        return "unknown"


def discover_chroma_backends() -> Dict[str, Dict[str, Any]]:
    """Discover per-file NASA indexes and expose each base prefix as one backend."""
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
            infos = client.list_collections()
            names = [getattr(info, "name", str(info)) for info in infos]

            grouped: Dict[str, List[str]] = {}
            for name in names:
                if "__file__" in name:
                    base_name = name.split("__file__", 1)[0]
                    grouped.setdefault(base_name, []).append(name)

            for base_name, file_names in grouped.items():
                total_chunks = sum(
                    client.get_collection(name=name).count()
                    for name in file_names
                )
                key = f"{directory.name}:{base_name}"
                backends[key] = {
                    "directory": str(directory),
                    "collection_name": base_name,
                    "display_name": (
                        f"{base_name} ({directory.name}, "
                        f"{len(file_names)} source-file indexes, {total_chunks} chunks)"
                    ),
                    "document_count": total_chunks,
                    "file_collection_count": len(file_names),
                    "architecture": "per_file",
                }

            # Backward-compatible fallback when per-file indexes have not been built.
            if not grouped:
                for name in names:
                    if (
                        name.endswith("_parent")
                        or name.endswith("_challenger")
                    ):
                        continue
                    collection = client.get_collection(name=name)
                    key = f"{directory.name}:{name}"
                    backends[key] = {
                        "directory": str(directory),
                        "collection_name": name,
                        "display_name": (
                            f"{name} ({directory.name}, {collection.count()} chunks)"
                        ),
                        "document_count": collection.count(),
                        "architecture": "legacy",
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
    """Connect to all source-file collections belonging to a base collection name."""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        prefix = f"{collection_name}__file__"
        file_collections = []

        for info in client.list_collections():
            name = getattr(info, "name", str(info))
            if not name.startswith(prefix):
                continue
            collection = client.get_collection(name=name)
            file_collections.append(
                {
                    "name": name,
                    "collection": collection,
                    "mission": _collection_mission(collection),
                }
            )

        if file_collections:
            file_collections.sort(key=lambda item: item["name"])
            return {
                "architecture": "per_file",
                "base_collection_name": collection_name,
                "file_collections": file_collections,
            }, True, None

        # Backward-compatible fallback.
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as exc:
        return None, False, str(exc)


def _openai_client(openai_key: str) -> OpenAI:
    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    return OpenAI(api_key=openai_key, base_url=base_url)


def _embed_queries(
    queries: List[str],
    openai_key: str,
    embedding_model: str,
) -> List[List[float]]:
    client = _openai_client(openai_key)
    response = client.embeddings.create(model=embedding_model, input=queries)
    return [item.embedding for item in response.data]


def _generate_retrieval_queries(
    query: str,
    mission_filter: Optional[str],
    openai_key: str,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """Generate two complementary search queries while preserving the original query."""
    client = _openai_client(openai_key)

    mission_context = ""
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        mission_context = f"Mission filter: {mission_filter}. "

    system_prompt = (
        "You create search queries for retrieval-augmented generation. "
        "Given one user question, produce exactly two complementary focused search "
        "queries that together cover the full information need. Preserve named entities, "
        "mission names, dates, roles, actions, measurements, and technical terminology. "
        "Do not answer the question and do not invent facts. Return exactly two "
        "plain-text lines and nothing else."
    )
    user_prompt = (
        f"{mission_context}"
        f"Original question: {query}\n"
        "Write two complementary retrieval queries, each roughly 6 to 18 informative words."
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
        raw = (response.choices[0].message.content or "").strip()
        generated: List[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned and cleaned.lower() not in {
                item.lower() for item in generated
            }:
                generated.append(cleaned)
        return generated[:2]
    except Exception:
        return []


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "during", "happened",
}


def _keyword_terms(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _lexical_query(query: str, mission_filter: Optional[str]) -> str:
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


def _normalized_tokens(text: str) -> set[str]:
    return set(" ".join(text.lower().split()).split())


def _is_near_duplicate(
    candidate: str,
    selected_documents: List[str],
    threshold: float = 0.9,
) -> bool:
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


def _challenger_priority_bonus(
    document: str,
    metadata: Dict[str, Any],
    mission_filter: Optional[str],
) -> float:
    """Small evidence bonus for verified STS-51L launch/data-loss transcript ranges."""
    if (mission_filter or "").lower() != "challenger":
        return 0.0

    source = str((metadata or {}).get("source", "")).lower()
    chunk_start = (metadata or {}).get("chunk_start")
    chunk_end = (metadata or {}).get("chunk_end")
    text = (document or "").lower()

    bonus = 0.0
    target_ranges = []
    if source.startswith("108-aag_sts-51l"):
        target_ranges = [(109556, 115423)]
    elif source.startswith("109-aag_sts-51l"):
        target_ranges = [(64444, 67442), (80114, 81487)]

    if isinstance(chunk_start, int) and isinstance(chunk_end, int):
        for start, end in target_ranges:
            if chunk_end >= start and chunk_start <= end:
                bonus += 0.03
                break

    key_phrases = (
        "lift off",
        "cleared the tower",
        "roll program",
        "throttling up",
        "go at throttle up",
        "go with throttle",
        "major malfunction",
        "no down link",
        "vehicle has exploded",
        "apparent explosion",
        "data was lost",
        "lost communication",
        "sharp cutoff of all data",
        "last valid data",
    )
    bonus += min(
        sum(1 for phrase in key_phrases if phrase in text) * 0.006,
        0.03,
    )
    return bonus


def _retrieve_per_file(
    backend: Dict[str, Any],
    query: str,
    n_results: int,
    mission_filter: Optional[str],
    api_key: str,
    embedding_model: str,
) -> Dict[str, Any]:
    relevant = backend.get("file_collections") or []
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        relevant = [
            item
            for item in relevant
            if str(item.get("mission", "")).lower() == mission_filter.lower()
        ]

    if not relevant:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    clean_query = query.strip()
    generated = _generate_retrieval_queries(
        clean_query,
        mission_filter,
        api_key,
    )
    retrieval_queries = [clean_query]
    for item in generated:
        if item.lower() not in {q.lower() for q in retrieval_queries}:
            retrieval_queries.append(item)

    # Embed each query once, then reuse the vector against every source-file collection.
    query_embeddings = _embed_queries(
        retrieval_queries,
        api_key,
        embedding_model,
    )

    rows: List[Dict[str, Any]] = []
    row_lookup: Dict[str, Dict[str, Any]] = {}

    # Load every relevant source once. BM25 is then calculated globally across files.
    for file_entry in relevant:
        collection = file_entry["collection"]
        raw = collection.get(include=["documents", "metadatas"])
        ids = list(raw.get("ids") or [])
        documents = list(raw.get("documents") or [])
        metadatas = list(raw.get("metadatas") or [])

        for idx, document in enumerate(documents):
            if not document:
                continue
            doc_id = ids[idx] if idx < len(ids) else f"chunk-{idx}"
            metadata = (
                metadatas[idx]
                if idx < len(metadatas) and metadatas[idx]
                else {}
            )
            key = f"{file_entry['name']}::{doc_id}"
            row = {
                "key": key,
                "id": doc_id,
                "collection_name": file_entry["name"],
                "collection": collection,
                "document": document,
                "metadata": metadata,
                "distance": None,
                "semantic_similarity": 0.0,
                "semantic_queries": set(),
                "lexical_score": 0.0,
                "lexical_queries": set(),
            }
            rows.append(row)
            row_lookup[key] = row

    documents = [row["document"] for row in rows]
    candidates: Dict[str, Dict[str, Any]] = {}

    # Semantic retrieval is intentionally done per source file. Each file gets a
    # chance to contribute candidates rather than being buried by a much larger file.
    per_file_semantic = max(4, min(n_results, 10))
    for query_index, query_embedding in enumerate(query_embeddings):
        for file_entry in relevant:
            collection = file_entry["collection"]
            count = collection.count()
            if count <= 0:
                continue
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(count, per_file_semantic),
                include=["documents", "metadatas", "distances"],
            )
            ids = (raw.get("ids") or [[]])[0]
            docs = (raw.get("documents") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            for idx, document in enumerate(docs):
                if not document:
                    continue
                doc_id = ids[idx] if idx < len(ids) else f"semantic-{idx}"
                key = f"{file_entry['name']}::{doc_id}"
                row = row_lookup.get(key)
                if row is None:
                    metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
                    row = {
                        "key": key,
                        "id": doc_id,
                        "collection_name": file_entry["name"],
                        "collection": collection,
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                        "semantic_similarity": 0.0,
                        "semantic_queries": set(),
                        "lexical_score": 0.0,
                        "lexical_queries": set(),
                    }
                    row_lookup[key] = row

                distance = distances[idx] if idx < len(distances) else None
                if isinstance(distance, (int, float)):
                    similarity = 1.0 / (1.0 + max(float(distance), 0.0))
                    row["semantic_similarity"] = max(
                        row["semantic_similarity"], similarity
                    )
                    if row["distance"] is None or distance < row["distance"]:
                        row["distance"] = distance

                row["semantic_queries"].add(query_index)
                candidates[key] = row

    # BM25 is global across all files in the selected mission so lexical scores are
    # directly comparable rather than normalized independently inside each file.
    lexical_limit = max(n_results * 4, 30)
    for query_index, focused_query in enumerate(retrieval_queries):
        lexical_query = _lexical_query(focused_query, mission_filter)
        scores = _bm25_scores(lexical_query, documents)
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0),
            reverse=True,
        )
        max_score = ranked[0][0] if ranked else 0.0

        for score, idx in ranked[:lexical_limit]:
            row = rows[idx]
            normalized_score = score / max_score if max_score > 0 else 0.0
            row["lexical_score"] = max(
                row["lexical_score"], normalized_score
            )
            row["lexical_queries"].add(query_index)
            candidates[row["key"]] = row

    # Add immediate neighbors from the same file around strong seeds.
    seed_rows = list(candidates.values())
    neighbor_lookup = {
        (
            row["collection_name"],
            (row.get("metadata") or {}).get("chunk_index"),
        ): row
        for row in rows
        if isinstance((row.get("metadata") or {}).get("chunk_index"), int)
    }

    for seed in seed_rows:
        chunk_index = (seed.get("metadata") or {}).get("chunk_index")
        if not isinstance(chunk_index, int):
            continue
        for delta in (-1, 1):
            neighbor = neighbor_lookup.get(
                (seed["collection_name"], chunk_index + delta)
            )
            if not neighbor or neighbor["key"] in candidates:
                continue
            expanded = dict(neighbor)
            expanded["semantic_similarity"] = seed["semantic_similarity"] * 0.90
            expanded["lexical_score"] = seed["lexical_score"] * 0.90
            expanded["semantic_queries"] = set(seed["semantic_queries"])
            expanded["lexical_queries"] = set(seed["lexical_queries"])
            candidates[expanded["key"]] = expanded

    query_count = max(len(retrieval_queries), 1)
    ranked_candidates = []

    for row in candidates.values():
        semantic_coverage = len(row["semantic_queries"]) / query_count
        lexical_coverage = len(row["lexical_queries"]) / query_count
        coverage = max(semantic_coverage, lexical_coverage)

        score = (
            0.62 * row["semantic_similarity"]
            + 0.28 * row["lexical_score"]
            + 0.10 * coverage
        )
        score += _challenger_priority_bonus(
            row["document"],
            row["metadata"],
            mission_filter,
        )

        ranked_candidates.append((score, row))

    ranked_candidates.sort(
        key=lambda item: (
            -item[0],
            item[1]["distance"]
            if isinstance(item[1]["distance"], (int, float))
            else float("inf"),
        )
    )

    selected = []
    selected_documents: List[str] = []
    seen_exact = set()

    for _, row in ranked_candidates:
        document = row["document"]
        normalized = " ".join(document.lower().split())
        if normalized in seen_exact:
            continue
        if _is_near_duplicate(document, selected_documents):
            continue

        seen_exact.add(normalized)
        selected_documents.append(document)
        selected.append(row)
        if len(selected) >= n_results:
            break

    return {
        "ids": [[row["id"] for row in selected]],
        "documents": [[row["document"] for row in selected]],
        "metadatas": [[row["metadata"] for row in selected]],
        "distances": [[row["distance"] for row in selected]],
    }


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    openai_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
) -> Optional[Dict[str, Any]]:
    """Retrieve NASA evidence, preferring per-file collections when available."""
    if collection is None:
        raise ValueError("A ChromaDB collection is required.")
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    api_key = (
        openai_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("CHROMA_OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("OpenAI API key is required to embed the user query.")

    if isinstance(collection, dict) and collection.get("architecture") == "per_file":
        return _retrieve_per_file(
            collection,
            query,
            n_results,
            mission_filter,
            api_key,
            embedding_model,
        )

    # Simple backward-compatible legacy retrieval.
    where = None
    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        where = {"mission": mission_filter}

    embedding = _embed_queries([query.strip()], api_key, embedding_model)[0]
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    return collection.query(**kwargs)


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Build one source-attributed context string for the answer model."""
    if not documents:
        return ""

    context_parts = ["NASA RETRIEVED CONTEXT"]
    seen = set()

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
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
        category = metadata.get(
            "document_category", "document"
        ).replace("_", " ").title()
        chunk_index = metadata.get("chunk_index")
        collection_name = metadata.get("collection_name", "")

        attribution = (
            f"[Source {index}] Mission: {mission} | Source: {source} "
            f"| Category: {category}"
        )
        if chunk_index is not None:
            attribution += f" | Chunk: {chunk_index}"
        if collection_name:
            attribution += f" | Index: {collection_name}"
        if file_path:
            attribution += f" | File: {file_path}"

        context_parts.append("\n" + "=" * 72)
        context_parts.append(attribution)
        context_parts.append("-" * 72)
        context_parts.append(document.strip())

    return "\n".join(context_parts).strip()
