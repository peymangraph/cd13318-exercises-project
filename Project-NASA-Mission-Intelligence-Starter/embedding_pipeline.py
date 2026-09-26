#!/usr/bin/env python3
"""Build and inspect per-file ChromaDB indexes for the NASA mission corpus."""

import argparse
import hashlib
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _slug(value: str, max_length: int = 80) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return (value or "unknown")[:max_length].strip("_") or "unknown"


def file_collection_name(base_name: str, file_path: Path, mission: str) -> str:
    """Return a stable Chroma collection name for one source file."""
    path_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:8]
    source_slug = _slug(file_path.stem, max_length=70)
    return f"{base_name}__file__{_slug(mission, 24)}__{source_slug}_{path_hash}"


class ChromaEmbeddingPipelineTextOnly:
    """Create OpenAI embeddings for text chunks in one Chroma collection."""

    def __init__(
        self,
        openai_api_key: str | None,
        chroma_persist_directory: str = "./chroma_db_openai",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 400,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name

        base_url = (
            "https://openai.vocareum.com/v1"
            if openai_api_key and openai_api_key.startswith("voc")
            else None
        )
        self.openai_client = (
            OpenAI(api_key=openai_api_key, base_url=base_url)
            if openai_api_key
            else None
        )

        Path(chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "NASA source-file chunks",
                "embedding_model": embedding_model,
            },
        )

    def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into fixed-size character chunks with consistent overlap."""
        clean_text = text.strip()
        if not clean_text:
            return []

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            chunk = clean_text[start:end]

            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_size": len(chunk),
                    "configured_chunk_size": self.chunk_size,
                    "configured_chunk_overlap": self.chunk_overlap,
                    "content_sha256": hashlib.sha256(
                        chunk.encode("utf-8")
                    ).hexdigest(),
                }
            )
            chunks.append((chunk, chunk_metadata))

            if end >= len(clean_text):
                break
            start += step
            chunk_index += 1

        total_chunks = len(chunks)
        for _, chunk_metadata in chunks:
            chunk_metadata["total_chunks"] = total_chunks

        return chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.openai_client:
            raise ValueError("An OpenAI API key is required to create embeddings.")
        if not texts:
            return []
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def extract_mission_from_path(file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        if "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        if "challenger" in path_str or "sts-51l" in path_str or "sts_51l" in path_str:
            return "challenger"
        return "unknown"

    @staticmethod
    def extract_data_type_from_path(file_path: Path) -> str:
        name = str(file_path).lower()
        if "audio" in name:
            return "audio_transcript"
        if "transcript" in name or "transscript" in name:
            return "transcript"
        if "flight_plan" in name:
            return "flight_plan"
        if "textract" in name:
            return "textract_extracted"
        return "document"

    @staticmethod
    def extract_document_category_from_filename(filename: str) -> str:
        value = filename.lower()
        if "pao" in value:
            return "public_affairs_officer"
        if "flight_plan" in value:
            return "flight_plan"
        if "mission_audio" in value:
            return "mission_audio"
        if "ntrs" in value:
            return "nasa_archive"
        if "tec" in value:
            return "technical"
        if "_cm" in value or value.startswith("as13_cm"):
            return "command_module"
        return "general_document"

    def process_text_file(
        self, file_path: Path
    ) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("Could not read %s: %s", file_path, exc)
            return []

        if not content.strip():
            return []

        metadata = {
            "source": file_path.stem,
            "file_path": str(file_path),
            "file_type": "text",
            "content_type": "full_text",
            "mission": self.extract_mission_from_path(file_path),
            "data_type": self.extract_data_type_from_path(file_path),
            "document_category": self.extract_document_category_from_filename(
                file_path.name
            ),
            "file_size": len(content),
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_name": self.collection_name,
        }
        return self.chunk_text(content, metadata)

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        if update_mode not in {"skip", "update", "replace"}:
            raise ValueError("update_mode must be skip, update, or replace.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        stats = {"added": 0, "updated": 0, "skipped": 0}
        if not documents:
            return stats

        existing = self.collection.get()
        existing_ids = set(existing.get("ids") or [])

        if update_mode == "replace" and existing_ids:
            self.collection.delete(ids=list(existing_ids))
            existing_ids.clear()

        def document_id(metadata: Dict[str, Any]) -> str:
            return f"chunk_{int(metadata.get('chunk_index', 0)):06d}"

        for offset in range(0, len(documents), batch_size):
            batch = documents[offset : offset + batch_size]
            add_rows = []
            update_rows = []

            for text, metadata in batch:
                doc_id = document_id(metadata)
                if doc_id in existing_ids and update_mode == "skip":
                    stats["skipped"] += 1
                    continue
                if doc_id in existing_ids and update_mode == "update":
                    update_rows.append((doc_id, text, metadata))
                else:
                    add_rows.append((doc_id, text, metadata))

            for rows, operation in ((add_rows, "add"), (update_rows, "update")):
                if not rows:
                    continue

                texts = [row[1] for row in rows]
                embeddings = self.get_embeddings(texts)
                ids = [row[0] for row in rows]
                metadatas = [row[2] for row in rows]

                if operation == "add":
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    stats["added"] += len(rows)
                else:
                    self.collection.update(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    stats["updated"] += len(rows)

        if update_mode == "update":
            generated_ids = {
                document_id(metadata) for _, metadata in documents
            }
            stale_ids = existing_ids - generated_ids
            if stale_ids:
                self.collection.delete(ids=list(stale_ids))

        return stats


def scan_text_files(base_path: str) -> List[Path]:
    base = Path(base_path)
    if (base / "data_text").is_dir():
        base = base / "data_text"

    files: List[Path] = []
    if base.name.lower() in {"apollo11", "apollo13", "challenger"} and base.is_dir():
        files.extend(base.rglob("*.txt"))
    else:
        for mission_dir in ("apollo11", "apollo13", "challenger"):
            directory = base / mission_dir
            if directory.is_dir():
                files.extend(directory.rglob("*.txt"))

    return sorted(
        p
        for p in files
        if not p.name.startswith(".") and "summary" not in p.name.lower()
    )


def list_file_collections(
    chroma_dir: str, base_collection_name: str
) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=chroma_dir)
    prefix = f"{base_collection_name}__file__"
    rows = []

    for info in client.list_collections():
        name = getattr(info, "name", str(info))
        if not name.startswith(prefix):
            continue
        collection = client.get_collection(name=name)
        first = collection.get(limit=1, include=["metadatas"])
        metadata = ((first.get("metadatas") or [{}])[:1] or [{}])[0] or {}
        rows.append(
            {
                "collection_name": name,
                "chunks": collection.count(),
                "mission": metadata.get("mission", "unknown"),
                "source": metadata.get("source", "unknown"),
                "file_path": metadata.get("file_path", ""),
                "chunk_size": metadata.get("configured_chunk_size"),
                "chunk_overlap": metadata.get("configured_chunk_overlap"),
            }
        )

    rows.sort(key=lambda row: (row["mission"], row["source"]))
    return rows


def verify_file_collection(
    client,
    collection_name: str,
) -> Dict[str, Any]:
    """Verify that a collection's persisted vector index can answer a query."""
    try:
        collection = client.get_collection(name=collection_name)
        sample = collection.get(limit=1, include=["embeddings"])
        ids = list(sample.get("ids") or [])
        embeddings = sample.get("embeddings")
        if not ids or embeddings is None or len(embeddings) == 0:
            return {
                "collection_name": collection_name,
                "ok": False,
                "error": "Collection has no retrievable sample embedding.",
            }

        embedding = embeddings[0]
        collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        return {"collection_name": collection_name, "ok": True}
    except Exception as exc:
        return {
            "collection_name": collection_name,
            "ok": False,
            "error": str(exc),
        }


def verify_per_file_indexes(
    chroma_dir: str,
    base_collection_name: str,
) -> Dict[str, Any]:
    """Check every per-file collection, including its on-disk HNSW segment."""
    client = chromadb.PersistentClient(path=chroma_dir)
    collections = list_file_collections(chroma_dir, base_collection_name)
    checks = [
        verify_file_collection(client, row["collection_name"])
        for row in collections
    ]
    broken = [row for row in checks if not row["ok"]]
    return {
        "file_collection_count": len(checks),
        "healthy": len(checks) - len(broken),
        "broken": len(broken),
        "checks": checks,
    }


def repair_broken_indexes(args: argparse.Namespace) -> Dict[str, Any]:
    """Rebuild only per-file collections whose persisted vector index is broken."""
    client = chromadb.PersistentClient(path=args.chroma_dir)
    rows = list_file_collections(args.chroma_dir, args.collection_name)
    row_by_name = {row["collection_name"]: row for row in rows}

    verification = verify_per_file_indexes(
        args.chroma_dir,
        args.collection_name,
    )
    broken_checks = [
        check for check in verification["checks"] if not check["ok"]
    ]

    repaired = []
    failed = []

    for check in broken_checks:
        name = check["collection_name"]
        row = row_by_name.get(name)
        if not row:
            failed.append(
                {
                    "collection_name": name,
                    "error": "Collection metadata could not be read for repair.",
                }
            )
            continue

        file_path = Path(row["file_path"])
        if not file_path.exists():
            failed.append(
                {
                    "collection_name": name,
                    "error": f"Source file not found: {file_path}",
                }
            )
            continue

        try:
            client.delete_collection(name=name)
            pipeline = ChromaEmbeddingPipelineTextOnly(
                openai_api_key=args.openai_key,
                chroma_persist_directory=args.chroma_dir,
                collection_name=name,
                embedding_model=args.embedding_model,
                chunk_size=int(row.get("chunk_size") or args.chunk_size),
                chunk_overlap=int(
                    row.get("chunk_overlap") or args.chunk_overlap
                ),
            )
            chunks = pipeline.process_text_file(file_path)
            result = pipeline.add_documents_to_collection(
                chunks,
                file_path=file_path,
                batch_size=args.batch_size,
                update_mode="replace",
            )
            post_check = verify_file_collection(client, name)
            if post_check["ok"]:
                repaired.append(
                    {
                        "collection_name": name,
                        "source": row.get("source"),
                        "chunks": len(chunks),
                        "added": result["added"],
                    }
                )
            else:
                failed.append(post_check)
        except Exception as exc:
            failed.append(
                {
                    "collection_name": name,
                    "error": str(exc),
                }
            )

    return {
        "broken_found": len(broken_checks),
        "repaired": repaired,
        "failed": failed,
    }


def build_per_file_indexes(args: argparse.Namespace) -> Dict[str, Any]:
    files = scan_text_files(args.data_path)
    if not files:
        raise FileNotFoundError(f"No NASA text files found under {args.data_path}")

    totals = {
        "files_processed": 0,
        "documents_added": 0,
        "documents_updated": 0,
        "documents_skipped": 0,
        "errors": 0,
        "total_chunks": 0,
        "missions": {},
        "collections": [],
    }

    for file_path in files:
        mission = ChromaEmbeddingPipelineTextOnly.extract_mission_from_path(file_path)
        collection_name = file_collection_name(
            args.collection_name, file_path, mission
        )
        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key=args.openai_key,
            chroma_persist_directory=args.chroma_dir,
            collection_name=collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        try:
            chunks = pipeline.process_text_file(file_path)
            result = pipeline.add_documents_to_collection(
                chunks,
                file_path=file_path,
                batch_size=args.batch_size,
                update_mode=args.update_mode,
            )
            totals["files_processed"] += 1
            totals["total_chunks"] += len(chunks)
            totals["documents_added"] += result["added"]
            totals["documents_updated"] += result["updated"]
            totals["documents_skipped"] += result["skipped"]

            mission_stats = totals["missions"].setdefault(
                mission,
                {"files": 0, "chunks": 0},
            )
            mission_stats["files"] += 1
            mission_stats["chunks"] += len(chunks)

            totals["collections"].append(
                {
                    "collection_name": collection_name,
                    "mission": mission,
                    "source": file_path.stem,
                    "chunks": len(chunks),
                    "added": result["added"],
                    "updated": result["updated"],
                    "skipped": result["skipped"],
                }
            )
            logger.info(
                "Indexed %s -> %s (%s chunks)",
                file_path.name,
                collection_name,
                len(chunks),
            )
        except Exception as exc:
            totals["errors"] += 1
            logger.exception("Failed processing %s: %s", file_path, exc)

    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NASA per-file ChromaDB embedding pipeline"
    )
    parser.add_argument("--data-path", default="./data_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
    )
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repair-broken", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.verify_only:
        print(
            verify_per_file_indexes(
                args.chroma_dir,
                args.collection_name,
            )
        )
        return

    if args.repair_broken:
        if not args.openai_key:
            raise SystemExit(
                "OPENAI_API_KEY or --openai-key is required for --repair-broken."
            )
        print(repair_broken_indexes(args))
        return

    if args.stats_only:
        collections = list_file_collections(
            args.chroma_dir, args.collection_name
        )
        mission_counts = Counter(row["mission"] for row in collections)
        print(
            {
                "architecture": "per_file_collections",
                "file_collection_count": len(collections),
                "total_chunks": sum(row["chunks"] for row in collections),
                "missions": dict(mission_counts),
                "collections": collections,
            }
        )
        return

    if not args.openai_key:
        raise SystemExit("OPENAI_API_KEY or --openai-key is required.")

    start = time.time()
    stats = build_per_file_indexes(args)
    stats["architecture"] = "per_file_collections"
    stats["chunking"] = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    stats["elapsed_seconds"] = round(time.time() - start, 2)
    print(stats)


if __name__ == "__main__":
    main()
