# Large Language Models and Text Generation

Personal coursework and final-project repository maintained by **Peyman Mohammad Hassan (@peymangraph)** for the Udacity **Large Language Models and Text Generation** program.

This repository is a fork of the original Udacity course repository:

`https://github.com/udacity/cd13318-exercises-project.git`

All completion work is performed only in **`peymangraph/cd13318-exercises-project`**. The Udacity upstream repository is not modified.

## Upstream Synchronization

Before continuing the final project, this fork was verified against the latest Udacity `main`.

- Latest Udacity upstream commit checked: `4cb94352011993c322e057dccf78d83438667b0b`
- Latest upstream tree SHA: `943ab1634dcd759a94b7d097f1c275274e616477`
- Local synchronized `main` commit: `c51248fb241c7b1eafd6f8419ac48adc715f4c1c`
- Local `main` tree SHA: `943ab1634dcd759a94b7d097f1c275274e616477`
- The upstream and local `main` trees match exactly.
- The `final-project` branch is based on the synchronized local `main`.
- The unrelated upstream merge author is not present in the active `main` or `final-project` ancestry.
- New project commits are authored and committed by **@peymangraph**.

This approach keeps the repository current with Udacity while maintaining a clean working history for the final project.

## Current Project Status

The completed coursework and NASA Mission Intelligence project are now on the repository's **`main`** branch.

`main` is the canonical working/submission branch. The former `final-project` branch is no longer the primary branch.

### Completed

- All identified course starter exercises have been completed.
- NASA Mission Intelligence embedding pipeline is implemented.
- ChromaDB semantic retrieval and mission filtering are implemented.
- Grounded NASA-expert LLM integration is implemented.
- Streamlit chat application is implemented.
- RAGAS Response Relevancy and Faithfulness evaluation are implemented.
- A six-question NASA evaluation dataset is included.
- End-to-end batch evaluation tooling is included.
- Rubric-aligned project documentation is included.
- GitHub Actions syntax/completeness checks are included.
- Repository ownership metadata is configured for **@peymangraph** on the working branch.
- Latest Udacity upstream content has been synchronized before final validation.
- Runtime embedding validation completed successfully: 12 NASA source files produced 15,563 chunks with 0 errors.

### Remaining Before Final Merge

Complete these in order:

1. Run the Streamlit end-to-end smoke test against the populated ChromaDB.
2. Run the complete RAGAS batch evaluation and record metrics.
3. Perform the final submission audit on `main`.

The detailed completion checklist is maintained in **`TASKS.md`**.

> **Issue tracking note:** GitHub Issues are currently disabled for this repository. Until they are enabled under **Settings → General → Features → Issues**, `TASKS.md` is the active task tracker.

## NASA Mission Intelligence Final Project

The final project is located in:

`Project-NASA-Mission-Intelligence-Starter/`

It implements a complete Retrieval-Augmented Generation system over NASA documents from:

- **Apollo 11**
- **Apollo 13**
- **Challenger / STS-51L**

### Architecture

```text
NASA mission text files
        ↓
embedding_pipeline.py
        ↓
OpenAI embeddings
        ↓
ChromaDB
        ↓
rag_client.py
        ↓
Retrieved source-attributed context
        ↓
llm_client.py
        ↓
Grounded NASA mission answer
        ↓
ragas_evaluator.py
        ↓
Response Relevancy + Faithfulness
        ↓
chat.py / Streamlit
```

### Main Final-Project Files

- `embedding_pipeline.py` — configurable chunking, overlap, OpenAI embeddings, metadata, ChromaDB persistence, update modes, and statistics
- `rag_client.py` — semantic retrieval, configurable top-k, mission filtering, sorting/deduplication, and source-attributed context construction
- `llm_client.py` — NASA mission expert prompt, retrieved-context grounding, source citation instructions, uncertainty handling, and bounded conversation history
- `ragas_evaluator.py` — Response Relevancy and Faithfulness evaluation with validation/error handling
- `chat.py` — Streamlit user interface integrating retrieval, generation, mission selection, sources, and evaluation
- `batch_evaluate.py` — end-to-end batch evaluation with per-question and aggregate metrics
- `evaluation_dataset.txt` — six mission-relevant evaluation questions spanning overview, emergency, disaster analysis, crew, technical, and timeline categories
- `README.md` — project-specific setup instructions and rubric mapping

## Udacity Rubric Coverage

### Embedding & Data Pipeline

- Runtime-configurable `chunk_size` and `chunk_overlap`
- Chunks bounded by configured chunk size
- Consistent overlap between consecutive chunks
- OpenAI embeddings for indexed chunks
- Per-chunk source/filepath and mission metadata
- `skip`, `update`, and `replace` update modes
- Configurable ChromaDB directory and collection
- `--stats-only` collection statistics

### Retrieval & LLM Integration

- User-question embedding and ChromaDB similarity retrieval
- Runtime-configurable top-k
- Optional mission metadata filtering
- Score sorting and deduplication
- Clear context separators and source attribution
- NASA mission expert system prompt
- Bounded role/content conversation history
- Retrieved-context grounding and uncertainty handling

### Real-Time Evaluation

- Response Relevancy
- Faithfulness
- Structured evaluator output
- Clear handling of empty or malformed inputs
- Batch evaluation
- Per-question and aggregate metrics
- At least five mission-relevant evaluation questions across multiple categories

## Course Exercises

Completed exercise areas include:

- Applied Prompting and Inference
- Implementing Chatbot with LLM
- Implementing RAG with Vector Databases
- Tokens, Embeddings and Vector Search
- RAG Evaluation Implementation
- Strategic Model Selection and Economics

## Working With the Repository

Clone your fork:

```bash
git clone https://github.com/peymangraph/cd13318-exercises-project.git
cd cd13318-exercises-project
```

Use the canonical project branch:

```bash
git checkout main
git pull origin main
```

Open the NASA final project:

```bash
cd Project-NASA-Mission-Intelligence-Starter
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Set an OpenAI API key and follow the project-specific README for ChromaDB indexing, Streamlit launch, and batch evaluation.

## RAGAS Compatibility

The latest Udacity upstream includes the RAGAS 0.4.3 compatibility dependencies:

```text
langchain-google-vertexai==3.2.3
langchain-community==0.4.2
ragas==0.4.3
```

The upstream `fix_ragas.png` reference is retained in the NASA project directory, and the project-specific README includes the compatibility guidance.

## Pull Request

The former `final-project` work has been promoted directly to `main`. PR #1 is no longer required for branch promotion. Runtime validation tasks remain tracked in `TASKS.md`.

## Attribution

Udacity-provided starter code, course materials, datasets, and reference/solution files retain their original attribution and license.

New implementation work, repository maintenance, final-project integration, documentation, synchronization commits, and final-project commits in this fork are maintained under **Peyman Mohammad Hassan / @peymangraph**.
