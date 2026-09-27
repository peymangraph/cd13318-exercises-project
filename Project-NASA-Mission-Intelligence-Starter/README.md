# NASA Mission Intelligence — Final Project

Maintainer: **Peyman Mohammad Hassan (@peymangraph)**

This is the completed Udacity final-project implementation for a Retrieval-Augmented Generation (RAG) system over NASA material from **Apollo 11**, **Apollo 13**, and **Challenger (STS-51L)**.

## Frozen submission configuration

The submission baseline is intentionally frozen at:

- Child chunks: **400 characters / 100 overlap**
- Parent chunks: **1200 characters / 300 overlap**
- Challenger: **dedicated Challenger collection using 400 / 100**
- Embedding model: **text-embedding-3-small**
- Batch evaluation retrieval: **top-k 10**
- Retrieval quality gate: **none**
- Generation model default: **gpt-4o-mini**
- Answering behavior: direct, source-grounded, citation-required responses that omit unsupported details instead of appending generic noncommittal caveats

Do not change chunking, ranking, prompt behavior, or the evaluation questions for the submitted baseline.

## Architecture

1. `embedding_pipeline.py` reads NASA text files and builds three persistent ChromaDB collections:
   - `nasa_space_missions_text`: child chunks at 400 / 100
   - `nasa_space_missions_text_parent`: parent chunks at 1200 / 300
   - `nasa_space_missions_text_challenger`: Challenger-only child chunks at 400 / 100
2. `rag_client.py` uses hierarchical parent-child retrieval for Apollo questions. It combines semantic retrieval, BM25 lexical retrieval, parent-region guidance, neighbor expansion, reciprocal-rank-style scoring, and deduplication.
3. Challenger-filtered questions automatically route to the dedicated Challenger collection.
4. `llm_client.py` generates concise grounded answers from retrieved NASA context, requires inline source labels, preserves technical wording and measurements from evidence, and omits unsupported details.
5. `ragas_evaluator.py` computes **Response Relevancy** and **Faithfulness** with RAGAS.
6. `chat.py` provides the Streamlit chat interface.
7. `batch_evaluate.py` runs the fixed evaluation dataset end-to-end and optionally re-scores the exact same generated answers multiple times for evaluator-stability diagnostics.
8. `evaluation_dataset.txt` contains six fixed mission-relevant questions across overview, emergency, disaster analysis, crew, technical, and timeline categories.

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

## Upstream RAGAS 0.4.3 compatibility note

The required packages are already included in `requirements.txt`, including:

```text
langchain-google-vertexai==3.2.3
langchain-community==0.4.2
ragas==0.4.3
```

If the Udacity workspace raises a VertexAI import error inside RAGAS 0.4.3, use the compatibility fix illustrated by `fix_ragas.png`.

## 1. Build the ChromaDB indexes

Run from this project directory:

    python embedding_pipeline.py \
      --data-path ./data_text \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --chunk-size 400 \
      --chunk-overlap 100 \
      --parent-chunk-size 1200 \
      --parent-chunk-overlap 300 \
      --update-mode replace

The runtime options satisfy the project requirements:

- `--chunk-size`
- `--chunk-overlap`
- `--parent-chunk-size`
- `--parent-chunk-overlap`
- `--challenger-collection-name`
- `--chroma-dir`
- `--collection-name`
- `--embedding-model`
- `--batch-size`
- `--update-mode skip|update|replace`

The pipeline stores per-chunk metadata including source, filepath, mission, chunk index, chunk boundaries, and content hash.

## 2. Inspect collection statistics

This mode does not require an OpenAI API call:

    python embedding_pipeline.py \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --stats-only

The final verified index shape was:

- child: **20,749 chunks**
- parent: **6,917 chunks**
- Challenger-only: **1,268 chunks**
- source documents: **12 total**, including **3 Challenger documents**

## 3. Launch the chat application

    streamlit run chat.py

The sidebar supports:

- ChromaDB collection selection
- OpenAI model selection
- mission filtering: all missions, Apollo 11, Apollo 13, Challenger
- runtime top-k retrieval
- optional RAGAS evaluation

Each answer is instructed to rely on retrieved NASA evidence and cite source labels inline.

## 4. Run batch evaluation

Final baseline:

    python batch_evaluate.py \
      --dataset evaluation_dataset.txt \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --top-k 10

Optional evaluator-stability diagnostic using the exact same generated answer and retrieved contexts:

    python batch_evaluate.py \
      --dataset evaluation_dataset.txt \
      --chroma-dir ./chroma_db_openai \
      --collection-name nasa_space_missions_text \
      --top-k 10 \
      --repeat-evaluations 3

The batch runner:

- loads the fixed evaluation dataset,
- retrieves mission-filtered NASA chunks,
- generates one grounded answer per question,
- computes Response Relevancy and Faithfulness,
- optionally repeats only the RAGAS scoring stage,
- prints per-question means, standard deviations, and aggregate metrics.

## Final verified evaluation

Three repeated RAGAS scores per generated answer produced:

| Category | Mission | Mean Response Relevancy | Mean Faithfulness |
|---|---|---:|---:|
| overview | Apollo 11 | 0.8806 | 0.9667 |
| emergency | Apollo 13 | 0.9015 | 0.8333 |
| disaster analysis | Challenger | 0.7574 | 0.9487 |
| crew | Apollo 11 | 0.9846 | 1.0000 |
| technical | Apollo 13 | 0.8973 | 1.0000 |
| timeline | Challenger | 0.7579 | 0.7949 |

Aggregate:

- **Mean Response Relevancy: 0.8632**
- **Mean Faithfulness: 0.9239**

RAGAS uses an LLM-based evaluator, so small run-to-run variation is expected.

## Evaluation dataset

`evaluation_dataset.txt` uses one JSON object per line and contains six fixed questions. The questions must not be rewritten for metric optimization.

Categories:

- overview
- emergency
- disaster analysis
- crew
- technical
- timeline

## Rubric mapping

### Embedding & Data Pipeline

- Runtime-configurable chunk size and overlap: implemented.
- Child chunks do not exceed configured chunk size and use consistent overlap.
- Hierarchical parent chunks are separately configurable.
- OpenAI embedding model is used for every indexed chunk.
- Source/filepath and mission metadata are stored per chunk.
- `skip`, `update`, and `replace` modes are supported.
- Persistent ChromaDB directory and collection names are configurable.
- `--stats-only` reports collection size and aggregate metadata.
- Dedicated Challenger collection is built automatically.

### Retrieval & LLM Integration

- User and reformulated search queries are explicitly embedded before semantic search.
- Apollo retrieval uses child semantic/BM25 evidence with parent-region guidance.
- Challenger-filtered retrieval routes to the dedicated Challenger collection.
- Runtime top-k retrieval is implemented.
- Mission metadata filtering is implemented.
- Results are score-sorted and deduplicated.
- Retrieved context uses clear separators and source attributions.
- System prompt identifies the assistant as a NASA mission expert and requires source citations.
- Conversation history is retained as role/content turns with bounded history.
- The model is instructed to rely only on retrieved context, omit unsupported details, and state insufficiency only when no substantive answer can be supported.

### Real-Time Evaluation

- Response Relevancy: implemented.
- Faithfulness: implemented.
- Evaluator accepts question, retrieved contexts, and answer and returns structured metrics.
- Empty/malformed inputs return clear non-crashing error dictionaries.
- Batch evaluation is implemented.
- Repeat-evaluation diagnostic mode reports all scores, per-question means, and standard deviations without changing retrieval or generation.
- Evaluation dataset contains six mission-relevant questions across multiple required categories.

## Submission contents

The submission archive should contain this project directory, including:

- `embedding_pipeline.py`
- `rag_client.py`
- `llm_client.py`
- `ragas_evaluator.py`
- `chat.py`
- `batch_evaluate.py`
- `evaluation_dataset.txt`
- `requirements.txt`
- `README.md`
- `SUBMISSION_REPORT.md`
- `fix_ragas.png`
- `data_text/`

The generated `chroma_db_openai/` directory is intentionally not required in the archive because the indexes are reproducible from the supplied source data and documented command.

## Repository note

Udacity starter/reference materials retain their original attribution and license. The completed working implementation and repository maintenance are maintained in @peymangraph's repository.
