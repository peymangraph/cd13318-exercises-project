# NASA Mission Intelligence — Submission Report

## Project summary

This project implements an end-to-end Retrieval-Augmented Generation system over NASA mission material for Apollo 11, Apollo 13, and Challenger (STS-51L). It includes ingestion and embedding, persistent ChromaDB indexes, hierarchical retrieval, mission filtering, grounded LLM generation, a Streamlit chat interface, and real-time RAGAS evaluation.

## Frozen final configuration

The submission baseline is frozen at:

- OpenAI embedding model: `text-embedding-3-small`
- Apollo child chunks: 400 characters with 100-character overlap
- Apollo parent chunks: 1200 characters with 300-character overlap
- Challenger collection: dedicated 400 / 100 child index
- Batch evaluation: top-k 10
- Generation model default: `gpt-4o-mini`
- No minimum cosine-similarity cutoff
- Direct source-grounded answering with inline source citations

## Index verification

The final local index statistics were:

- Main child collection: 20,749 chunks
- Parent collection: 6,917 chunks
- Challenger-only collection: 1,268 chunks
- Total source documents represented: 12
- Challenger source documents: 3

The project does not require the generated ChromaDB directory to be submitted because the indexes can be reproduced from the supplied `data_text/` directory and `embedding_pipeline.py`.

## Retrieval design

Apollo queries use a hierarchical parent-child retrieval flow. The parent index identifies promising regions while the child index supplies precise evidence. Candidate evidence combines semantic retrieval, BM25 lexical relevance, parent-region guidance, neighboring chunks, score fusion, and deduplication.

Challenger-filtered questions route to a separate Challenger collection to reduce competition from Apollo material while preserving the same source-grounded answering pipeline.

## Grounded generation

The LLM is instructed to use only retrieved NASA context and cite source labels inline. Unsupported details are omitted. The model states that evidence is insufficient only when no substantive answer can be supported, avoiding generic noncommittal endings that can distort Response Relevancy evaluation.

## Evaluation dataset

The fixed six-question dataset covers:

1. Apollo 11 mission overview
2. Apollo 13 emergency response
3. Challenger disaster sequence
4. Apollo 11 crew roles
5. Apollo 13 technical systems and procedures
6. Challenger final mission timeline

The evaluation questions were kept unchanged during final testing.

## Final evaluation results

A final stability run generated one answer per question and re-scored that exact answer three times with RAGAS.

| Category | Mission | Mean Response Relevancy | Mean Faithfulness |
|---|---|---:|---:|
| Overview | Apollo 11 | 0.8806 | 0.9667 |
| Emergency | Apollo 13 | 0.9015 | 0.8333 |
| Disaster analysis | Challenger | 0.7574 | 0.9487 |
| Crew | Apollo 11 | 0.9846 | 1.0000 |
| Technical | Apollo 13 | 0.8973 | 1.0000 |
| Timeline | Challenger | 0.7579 | 0.7949 |

Aggregate metrics:

- Mean Response Relevancy: **0.8632**
- Mean Faithfulness: **0.9239**

The repeated scoring diagnostic showed that the final response-relevancy results were stable across repeated RAGAS evaluations.

## Submission checklist

- [x] `embedding_pipeline.py` completed
- [x] `rag_client.py` completed
- [x] `llm_client.py` completed
- [x] `ragas_evaluator.py` completed
- [x] `chat.py` completed
- [x] `batch_evaluate.py` completed
- [x] `evaluation_dataset.txt` contains six mission-relevant questions
- [x] `requirements.txt` included
- [x] `README.md` includes setup, indexing, evaluation, and rubric mapping
- [x] NASA source text is included under `data_text/`
- [x] No remaining TODO / YOUR CODE HERE / NotImplementedError markers were found in the project directory
- [x] Final retrieval and prompt configuration frozen

## Reproduction commands

Build indexes:

```bash
python embedding_pipeline.py \
  --data-path ./data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --chunk-size 400 \
  --chunk-overlap 100 \
  --parent-chunk-size 1200 \
  --parent-chunk-overlap 300 \
  --update-mode replace
```

Inspect stats:

```bash
python embedding_pipeline.py \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --stats-only
```

Run the final batch baseline:

```bash
python batch_evaluate.py \
  --dataset evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --top-k 10
```

Run the three-pass evaluator stability check:

```bash
python batch_evaluate.py \
  --dataset evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --top-k 10 \
  --repeat-evaluations 3
```

Launch the app:

```bash
streamlit run chat.py
```
