#!/usr/bin/env python3
"""Build and inspect a ChromaDB index for the NASA mission text corpus."""

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


class ChromaEmbeddingPipelineTextOnly:
    """Create OpenAI embeddings for NASA text chunks and persist them in ChromaDB."""

    def __init__(
        self,
        openai_api_key: str | None,
        chroma_persist_directory: str = "./chroma_db_openai",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
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
        base_url = "https://openai.vocareum.com/v1" if openai_api_key and openai_api_key.startswith("voc") else None
        self.openai_client = (
            OpenAI(api_key=openai_api_key, base_url=base_url) if openai_api_key else None
        )

        Path(chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "NASA Apollo 11, Apollo 13, and Challenger text chunks",
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
                    "content_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
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

    def check_document_exists(self, doc_id: str) -> bool:
        result = self.collection.get(ids=[doc_id])
        return bool(result.get("ids"))

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]

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
    def _slug(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_") or "unknown"

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Generate a stable ID using mission, source, path hash, and chunk index."""
        mission = self._slug(str(metadata.get("mission", "unknown")))
        source = self._slug(str(metadata.get("source", file_path.stem)))
        path_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
        chunk_index = int(metadata.get("chunk_index", 0))
        return f"{mission}_{source}_{path_hash}_chunk_{chunk_index:04d}"

    def extract_mission_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        if "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        if "challenger" in path_str or "sts-51l" in path_str or "sts_51l" in path_str:
            return "challenger"
        return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
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

    def extract_document_category_from_filename(self, filename: str) -> str:
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

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
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
            "document_category": self.extract_document_category_from_filename(file_path.name),
            "file_size": len(content),
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self.chunk_text(content, metadata)

    def scan_text_files_only(self, base_path: str) -> List[Path]:
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
            p for p in files
            if not p.name.startswith(".") and "summary" not in p.name.lower()
        )

    def get_file_documents(self, file_path: Path) -> List[str]:
        result = self.collection.get(where={"file_path": str(file_path)})
        return list(result.get("ids") or [])

    def delete_documents_by_source(self, source_pattern: str) -> int:
        all_docs = self.collection.get()
        ids_to_delete = []
        for doc_id, metadata in zip(all_docs.get("ids", []), all_docs.get("metadatas", [])):
            if source_pattern.lower() in str((metadata or {}).get("source", "")).lower():
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

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

        existing_ids = set(self.get_file_documents(file_path))
        if update_mode == "replace" and existing_ids:
            self.collection.delete(ids=list(existing_ids))
            existing_ids.clear()

        generated_ids = {
            self.generate_document_id(file_path, metadata)
            for _, metadata in documents
        }

        for offset in range(0, len(documents), batch_size):
            batch = documents[offset : offset + batch_size]
            add_rows = []
            update_rows = []

            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)
                exists = doc_id in existing_ids or self.check_document_exists(doc_id)

                if exists and update_mode == "skip":
                    stats["skipped"] += 1
                    continue
                if exists and update_mode == "update":
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
            stale_ids = existing_ids - generated_ids
            if stale_ids:
                self.collection.delete(ids=list(stale_ids))

        return stats

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip",
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        files = self.scan_text_files_only(base_path)
        logger.info("Found %s NASA text files.", len(files))

        for file_path in files:
            mission = self.extract_mission_from_path(file_path)
            mission_stats = stats["missions"].setdefault(
                mission,
                {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0},
            )
            try:
                chunks = self.process_text_file(file_path)
                result = self.add_documents_to_collection(
                    chunks,
                    file_path=file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )
                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["documents_added"] += result["added"]
                stats["documents_updated"] += result["updated"]
                stats["documents_skipped"] += result["skipped"]

                mission_stats["files"] += 1
                mission_stats["chunks"] += len(chunks)
                mission_stats["added"] += result["added"]
                mission_stats["updated"] += result["updated"]
                mission_stats["skipped"] += result["skipped"]
            except Exception as exc:
                stats["errors"] += 1
                logger.exception("Failed processing %s: %s", file_path, exc)

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "persist_directory": self.chroma_persist_directory,
            "embedding_model": self.embedding_model,
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty.")
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        query_embedding = self.get_embedding(query_text)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        all_docs = self.collection.get()
        metadatas = all_docs.get("metadatas") or []
        file_paths = {
            (metadata or {}).get("file_path")
            for metadata in metadatas
            if (metadata or {}).get("file_path")
        }
        return {
            "collection_name": self.collection.name,
            "total_chunks": self.collection.count(),
            "unique_documents": len(file_paths),
            "missions": dict(Counter((m or {}).get("mission", "unknown") for m in metadatas)),
            "data_types": dict(Counter((m or {}).get("data_type", "unknown") for m in metadatas)),
            "document_categories": dict(
                Counter((m or {}).get("document_category", "unknown") for m in metadatas)
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA ChromaDB embedding pipeline")
    parser.add_argument("--data-path", default="./data_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--challenger-collection-name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--test-query")
    parser.add_argument("--delete-source")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.stats_only and not args.openai_key:
        raise SystemExit("OPENAI_API_KEY or --openai-key is required unless --stats-only is used.")

    challenger_collection_name = (
        args.challenger_collection_name or f"{args.collection_name}_challenger"
    )

    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    challenger_pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=challenger_collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.delete_source:
        print(
            {
                "main_deleted_chunks": pipeline.delete_documents_by_source(args.delete_source),
                "challenger_deleted_chunks": challenger_pipeline.delete_documents_by_source(
                    args.delete_source
                ),
            }
        )
        return

    if args.stats_only:
        print(
            {
                "main": pipeline.get_collection_stats(),
                "challenger": challenger_pipeline.get_collection_stats(),
            }
        )
        return

    start = time.time()
    main_stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode,
        batch_size=args.batch_size,
    )
    challenger_path = Path(args.data_path)
    if challenger_path.name.lower() != "challenger":
        challenger_path = challenger_path / "challenger"
    challenger_stats = challenger_pipeline.process_all_text_data(
        str(challenger_path),
        update_mode=args.update_mode,
        batch_size=args.batch_size,
    )

    stats = {
        "main": main_stats,
        "challenger": challenger_stats,
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }
    print(stats)

    if args.test_query:
        print(
            {
                "main": pipeline.query_collection(args.test_query, n_results=5),
                "challenger": challenger_pipeline.query_collection(
                    args.test_query, n_results=5
                ),
            }
        )


if __name__ == "__main__":
    main()
