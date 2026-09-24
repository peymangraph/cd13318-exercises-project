#!/usr/bin/env python3
"""Batch evaluation runner for the NASA Mission Intelligence RAG system."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import llm_client
import rag_client
import ragas_evaluator


def load_evaluation_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines test cases from evaluation_dataset.txt."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    cases: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

        for required in ("category", "mission", "question"):
            if not item.get(required):
                raise ValueError(f"Line {line_number} is missing required field: {required}")
        cases.append(item)

    if not cases:
        raise ValueError("Evaluation dataset contains no test cases.")
    return cases


def run_batch(
    cases: List[Dict[str, Any]],
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
    model: str,
    top_k: int,
) -> Dict[str, Any]:
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Could not initialize ChromaDB: {error}")

    per_question = []
    for index, case in enumerate(cases, start=1):
        retrieval = rag_client.retrieve_documents(
            collection,
            case["question"],
            n_results=top_k,
            mission_filter=case.get("mission"),
            openai_key=openai_key,
        )
        documents = (retrieval.get("documents") or [[]])[0]
        metadatas = (retrieval.get("metadatas") or [[]])[0]
        context = rag_client.format_context(documents, metadatas)

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=case["question"],
            context=context,
            conversation_history=[],
            model=model,
        )
        metrics = ragas_evaluator.evaluate_response_quality(
            case["question"],
            answer,
            documents,
            openai_key=openai_key,
        )

        result = {
            "index": index,
            "category": case["category"],
            "mission": case["mission"],
            "question": case["question"],
            "answer": answer,
            "metrics": metrics,
        }
        per_question.append(result)
        print(json.dumps(result, indent=2))

    aggregate: Dict[str, float] = {}
    for metric_name in ("response_relevancy", "faithfulness"):
        values = [
            item["metrics"][metric_name]
            for item in per_question
            if isinstance(item.get("metrics"), dict)
            and isinstance(item["metrics"].get(metric_name), (int, float))
        ]
        if values:
            aggregate[f"mean_{metric_name}"] = mean(values)

    summary = {
        "question_count": len(per_question),
        "aggregate_metrics": aggregate,
        "results": per_question,
    }
    print("\nBATCH SUMMARY")
    print(json.dumps({"question_count": summary["question_count"], "aggregate_metrics": aggregate}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch evaluate the NASA RAG system.")
    parser.add_argument("--dataset", default="evaluation_dataset.txt")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.openai_key:
        raise SystemExit("OPENAI_API_KEY or --openai-key is required.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    cases = load_evaluation_dataset(args.dataset)
    run_batch(
        cases=cases,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        openai_key=args.openai_key,
        model=args.model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
