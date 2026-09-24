#!/usr/bin/env python3
"""Streamlit chat application for NASA Mission Intelligence."""

import os
from typing import Dict, List, Optional

import streamlit as st

import llm_client
import rag_client
import ragas_evaluator

try:
    from ragas import SingleTurnSample  # noqa: F401
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

st.set_page_config(
    page_title="NASA Mission Intelligence",
    page_icon="🚀",
    layout="wide",
)


def discover_chroma_backends() -> Dict:
    return rag_client.discover_chroma_backends()


def initialize_rag_system(chroma_dir: str, collection_name: str):
    return rag_client.initialize_rag_system(chroma_dir, collection_name)


def retrieve_documents(
    collection,
    query: str,
    n_results: int,
    mission_filter: Optional[str],
    openai_key: str,
):
    return rag_client.retrieve_documents(
        collection,
        query,
        n_results=n_results,
        mission_filter=mission_filter,
        openai_key=openai_key,
    )


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    return rag_client.format_context(documents, metadatas)


def display_evaluation_metrics(scores: Dict):
    st.sidebar.subheader("📊 Response Quality")
    if not scores:
        st.sidebar.info("No evaluation has been run yet.")
        return
    if "error" in scores:
        st.sidebar.error(scores["error"])
        return

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(metric_name.replace("_", " ").title(), f"{score:.3f}")
            st.sidebar.progress(max(0.0, min(float(score), 1.0)))


def display_sources(result: Dict):
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    if not documents:
        return

    with st.expander("Retrieved NASA sources"):
        for index, (_, metadata) in enumerate(zip(documents, metadatas), start=1):
            metadata = metadata or {}
            distance = distances[index - 1] if index - 1 < len(distances) else None
            mission = metadata.get("mission", "unknown").replace("_", " ").title()
            source = metadata.get("source", "unknown")
            file_path = metadata.get("file_path", "")
            chunk_index = metadata.get("chunk_index", "n/a")
            st.markdown(f"**Source {index}: {source}**")
            st.caption(
                f"Mission: {mission} · Chunk: {chunk_index}"
                + (f" · Distance: {distance:.4f}" if isinstance(distance, (int, float)) else "")
            )
            if file_path:
                st.code(file_path, language=None)


def main():
    st.title("🚀 NASA Mission Intelligence")
    st.markdown(
        "Ask source-grounded questions about **Apollo 11**, **Apollo 13**, "
        "and **Challenger (STS-51L)**."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:
        st.header("Configuration")

        backends = discover_chroma_backends()
        usable_backends = {
            key: value
            for key, value in backends.items()
            if value.get("collection_name") and not value.get("error")
        }

        if not usable_backends:
            st.error("No usable ChromaDB collection was found.")
            st.info(
                "Run the embedding pipeline first, for example:\n\n"
                "python embedding_pipeline.py --data-path ./data_text"
            )
            st.stop()

        backend_key = st.selectbox(
            "Document collection",
            options=list(usable_backends.keys()),
            format_func=lambda key: usable_backends[key]["display_name"],
        )
        backend = usable_backends[backend_key]

        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        if not openai_key:
            st.warning("Enter an OpenAI API key to continue.")
            st.stop()

        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

        model = st.selectbox(
            "Generation model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            index=0,
        )

        mission_label = st.selectbox(
            "Mission filter",
            ["All missions", "Apollo 11", "Apollo 13", "Challenger"],
        )
        mission_map = {
            "All missions": None,
            "Apollo 11": "apollo_11",
            "Apollo 13": "apollo_13",
            "Challenger": "challenger",
        }
        mission_filter = mission_map[mission_label]

        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=5)
        enable_evaluation = st.checkbox(
            "Enable RAGAS evaluation",
            value=RAGAS_AVAILABLE,
            disabled=not RAGAS_AVAILABLE,
        )

        if st.session_state.last_evaluation and enable_evaluation:
            display_evaluation_metrics(st.session_state.last_evaluation)

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.session_state.last_result = None
            st.rerun()

    collection, success, error = initialize_rag_system(
        backend["directory"],
        backend["collection_name"],
    )
    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a NASA mission..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving NASA evidence and generating answer..."):
                try:
                    result = retrieve_documents(
                        collection,
                        prompt,
                        n_results=top_k,
                        mission_filter=mission_filter,
                        openai_key=openai_key,
                    )
                    documents = (result.get("documents") or [[]])[0]
                    metadatas = (result.get("metadatas") or [[]])[0]
                    context = format_context(documents, metadatas)

                    answer = llm_client.generate_response(
                        openai_key=openai_key,
                        user_message=prompt,
                        context=context,
                        conversation_history=st.session_state.messages[:-1],
                        model=model,
                    )
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    st.stop()

                st.markdown(answer)
                display_sources(result)

                st.session_state.last_result = result
                if enable_evaluation and RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        st.session_state.last_evaluation = (
                            ragas_evaluator.evaluate_response_quality(
                                question=prompt,
                                answer=answer,
                                contexts=documents,
                                openai_key=openai_key,
                            )
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


if __name__ == "__main__":
    main()
