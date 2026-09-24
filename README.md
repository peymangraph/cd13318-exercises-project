# Large Language Models and Text Generation

Personal coursework and final-project repository maintained by **Peyman Mohammad Hassan (@peymangraph)** for the Udacity **Large Language Models and Text Generation** program.

This repository is a fork of the original Udacity course repository. All new completion work is being performed only in **`peymangraph/cd13318-exercises-project`**. The upstream Udacity repository is not modified. This fork is synchronized to the latest upstream `main` content while intentionally avoiding the unrelated upstream merge commit `4cb9435` in the active branch ancestry.

## Current Project Status

Active work is on the **`final-project`** branch.

The `main` branch is synchronized to the latest upstream `main` snapshot through a local sync commit, while all project completion work remains on `final-project` until runtime validation and final review are complete.

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

### Remaining Before Final Merge

Complete these in order:

1. **P0-01 — Confirm GitHub Actions workflow execution**
2. **P0-02 — Run the full NASA embedding pipeline and populate ChromaDB**
3. **P0-03 — Run the Streamlit end-to-end smoke test**
4. **P0-04 — Run the complete RAGAS batch evaluation and record metrics**
5. **P0-05 — Review generated/cache artifacts before submission**
6. **P0-06 — Perform the final submission audit and merge `final-project` into `main`**

The detailed checklist is maintained in **`TASKS.md`**.

> **Issue tracking note:** GitHub Issues are currently disabled for this repository. Until they are enabled under **Settings → General → Features → Issues**, `TASKS.md` is the active completion tracker.

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

- `embedding_pipeline.py` — chunking, embeddings, metadata, ChromaDB persistence, update modes, statistics
- `rag_client.py` — semantic retrieval, configurable top-k, mission filtering, deduplication, context construction
- `llm_client.py` — NASA mission expert prompting, grounding, source citation instructions, bounded conversation history
- `ragas_evaluator.py` — Response Relevancy and Faithfulness evaluation
- `chat.py` — Streamlit chat application
- `batch_evaluate.py` — end-to-end batch evaluation
- `evaluation_dataset.txt` — six rubric-aligned NASA evaluation questions
- `README.md` — detailed setup instructions and rubric mapping

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

Switch to the active project branch:

```bash
git checkout final-project
git pull origin final-project
```

Open the NASA final project:

```bash
cd Project-NASA-Mission-Intelligence-Starter
```

Install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Set an OpenAI API key and follow the project-specific README for indexing, Streamlit launch, and batch evaluation.

## Pull Request

The current final-project work is tracked in the draft pull request:

**PR #1 — Finalize NASA Mission Intelligence project and complete course exercises**

The PR should remain unmerged until all runtime validation tasks are complete.

## Attribution

Udacity-provided starter code, course materials, datasets, and reference/solution files retain their original attribution and license.

New implementation work, repository maintenance, final-project integration, documentation, and commits in this fork are maintained under **Peyman Mohammad Hassan / @peymangraph**.
