# NASA Mission Intelligence — Final Project

Maintainer: **Peyman Mohammad Hassan (@peymangraph)**

This is the completed Udacity final-project implementation for a Retrieval-Augmented Generation (RAG) system over NASA material from **Apollo 11**, **Apollo 13**, and **Challenger (STS-51L)**.

## Architecture

1. `embedding_pipeline.py` reads NASA text files, chunks them, creates OpenAI embeddings, and persists them in ChromaDB.
2. `rag_client.py` embeds each user question, performs semantic retrieval, optionally filters by mission metadata, deduplicates/sorts results, and constructs source-attributed context.
3. `llm_client.py` generates a grounded answer using a NASA-expert system prompt, current retrieved context, and bounded conversation history.
4. `ragas_evaluator.py` computes **Response Relevancy** and **Faithfulness** using RAGAS.
5. `chat.py` provides the Streamlit interface.
6. `batch_evaluate.py` runs the evaluation dataset end-to-end and reports per-question plus aggregate scores.
7. `evaluation_dataset.txt` contains six mission-relevant evaluation questions across overview, emergency, disaster analysis, crew, technical, and timeline categories.

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies in `requirements.txt`

Install:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Windows PowerShell activation:

    .venv\Scripts\Activate.ps1

Set the API key:

Linux/macOS:

    export OPENAI_API_KEY="your-key"

PowerShell:

    $env:OPENAI_API_KEY="your-key"

## Upstream RAGAS 0.4.3 Compatibility Note

The fork has been synchronized with the latest Udacity `main` content as of upstream commit `4cb9435`. That upstream update added the `langchain-google-vertexai` and `langchain-community` dependencies plus a RAGAS 0.4.3 compatibility note.

The required packages are already included in `requirements.txt`:

```text
langchain-google-vertexai==3.2.3
langchain-community==0.4.2
ragas==0.4.3
```

If the Udacity workspace raises a VertexAI import error inside RAGAS 0.4.3, use the upstream compatibility fix by replacing the older `langchain_community` VertexAI imports in `ragas/llms/base.py` with the corresponding `langchain_google_vertexai` imports. The upstream reference image `fix_ragas.png` is retained in this project directory.

## 1. Build the ChromaDB index

Run from this project directory:

    python embedding_pipeline.py \
      --data-path ./data_text \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --chunk-size 500 \
      --chunk-overlap 100 \
      --update-mode replace

The runtime options satisfy the project rubric:

- `--chunk-size`
- `--chunk-overlap`
- `--chroma-dir`
- `--collection-name`
- `--embedding-model`
- `--batch-size`
- `--update-mode skip|update|replace`

The pipeline stores per-chunk metadata including `source`, `file_path`, `mission`, chunk index, chunk boundaries, and content hash.

## 2. Inspect collection statistics

This mode does not require an OpenAI API call:

    python embedding_pipeline.py \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --stats-only

The output includes total chunk count, unique source-document count, mission counts, data types, and document categories.

## 3. Launch the chat application

    streamlit run chat.py

The sidebar supports:

- ChromaDB collection selection
- OpenAI model selection
- mission filtering: all missions, Apollo 11, Apollo 13, Challenger
- runtime top-k retrieval
- optional RAGAS evaluation

Each answer is instructed to rely on retrieved NASA evidence, cite source labels, and explicitly acknowledge insufficient or conflicting context.

## 4. Run batch evaluation

    python batch_evaluate.py \
      --dataset evaluation_dataset.txt \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --top-k 5

The runner:

- loads the evaluation dataset,
- retrieves mission-filtered NASA chunks,
- generates a grounded answer,
- computes Response Relevancy and Faithfulness,
- prints results for each question,
- prints mean aggregate metrics.

## Evaluation dataset

`evaluation_dataset.txt` uses one JSON object per line. It includes six categories:

- overview
- emergency
- disaster analysis
- crew
- technical
- timeline

## Rubric mapping

### Embedding & Data Pipeline

- Runtime-configurable chunk size and overlap: implemented.
- Chunks never exceed configured chunk size: implemented.
- Consistent overlap between consecutive chunks: implemented with a fixed sliding window.
- OpenAI embedding model for every indexed chunk: implemented.
- Source/filepath and mission metadata: implemented.
- `skip`, `update`, and `replace` handling: implemented.
- Configurable persistent ChromaDB directory and collection: implemented.
- `--stats-only` with collection size and aggregates: implemented.

### Retrieval & LLM Integration

- User question is explicitly embedded with the OpenAI embedding model before Chroma similarity search.
- Runtime top-k retrieval: implemented.
- Mission metadata filtering: implemented.
- Results are score-sorted and deduplicated.
- LLM context uses separators and source attributions.
- System prompt identifies the assistant as a NASA mission expert and requires source citations.
- Conversation history is retained as role/content turns with bounded history.
- The model is instructed to rely on retrieved context and state uncertainty when evidence is insufficient.

### Real-Time Evaluation

- Response Relevancy: implemented.
- Faithfulness: implemented.
- Evaluator accepts question, retrieved contexts, and answer and returns structured metrics.
- Empty/malformed inputs return clear error dictionaries.
- Batch evaluation flow is implemented.
- Evaluation dataset contains at least five questions across multiple required categories.

## Repository note

Udacity starter/reference materials retain their original attribution and license. The completed working implementation and repository maintenance are maintained in @peymangraph's repository.
