"""Real-time RAGAS evaluation for NASA Mission Intelligence answers."""

import asyncio
import os
from typing import Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

try:
    from ragas import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


async def _score_sample(question: str, answer: str, contexts: List[str], api_key: str) -> Dict[str, float]:
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    response_relevancy = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    faithfulness = Faithfulness(llm=evaluator_llm)

    relevancy_score = await response_relevancy.single_turn_ascore(sample)
    faithfulness_score = await faithfulness.single_turn_ascore(sample)

    return {
        "response_relevancy": float(relevancy_score),
        "faithfulness": float(faithfulness_score),
    }


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    openai_key: str | None = None,
) -> Dict[str, float]:
    """Evaluate a (question, retrieved context, answer) triple.

    Returns a structured dictionary containing at least Response Relevancy and
    Faithfulness. Invalid inputs return a clear error dictionary rather than raising.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS is not available. Install dependencies from requirements.txt."}

    if not isinstance(question, str) or not question.strip():
        return {"error": "Evaluation requires a non-empty question."}
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "Evaluation requires a non-empty model answer."}
    if not isinstance(contexts, list) or not contexts or not any(
        isinstance(item, str) and item.strip() for item in contexts
    ):
        return {"error": "Evaluation requires at least one non-empty retrieved context."}

    clean_contexts = [item.strip() for item in contexts if isinstance(item, str) and item.strip()]
    api_key = openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY is required for RAGAS evaluation."}

    try:
        return asyncio.run(_score_sample(question.strip(), answer.strip(), clean_contexts, api_key))
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}
