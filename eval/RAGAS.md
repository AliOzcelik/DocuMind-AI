# RAGAS Evaluation Guide

This folder contains a RAGAS-ready dataset and an evaluation runner for this RAG app.

## Files

- `ragas_sample_dataset.csv`: golden sample dataset.
- `build_ragas_dataset.py`: regenerates the sample dataset from `ragas_data/*.pdf`.
- `ragas_evaluate.py`: runs this app's RAG pipeline, saves model responses, then runs RAGAS.

## Install Dependencies

RAGAS is not currently installed in the project venv. Install it first:

```bash
venv/bin/pip install ragas datasets
```

Make sure Ollama is running:

```bash
ollama serve
```

Make sure the PDFs used for the dataset are also ingested into the active Chroma collection. This dataset was generated from `ragas_data/*.pdf`; your app currently retrieves from the existing Chroma database built by the ingestion pipeline. If those PDFs are not in the indexed corpus, copy/ingest them before comparing scores.

## Run Evaluation

Quick smoke test on 5 questions:

```bash
venv/bin/python eval/ragas_evaluate.py --limit 5
```

Full run:

```bash
venv/bin/python eval/ragas_evaluate.py
```

Use a specific generator and judge model:

```bash
venv/bin/python eval/ragas_evaluate.py \
  --model qwen3.5:4b-mlx \
  --evaluator-model qwen3.5:9b-mlx
```

The script writes:

- `eval/ragas_llm_responses_<timestamp>.csv`: every question, reference answer, retrieved contexts, and LLM answer.
- `eval/ragas_scores_<timestamp>.csv`: row-level RAGAS scores.

It also prints the aggregate RAGAS result immediately in the terminal.

## Recommended Metrics

Default metrics:

```text
faithfulness
answer_relevancy
context_precision
context_recall
```

Use fewer metrics for faster debugging:

```bash
venv/bin/python eval/ragas_evaluate.py --limit 5 --metrics faithfulness,answer_relevancy
```

## How to Compare Hyperparameters

Use the same dataset for every run. Change one thing at a time, run `ragas_evaluate.py`, and compare the output CSVs.

Restart the Python process between config changes. Several modules create `Config()` and retriever caches at import time.

Good baseline command:

```bash
venv/bin/python eval/ragas_evaluate.py \
  --model qwen3.5:4b-mlx \
  --responses-out eval/baseline_responses.csv \
  --scores-out eval/baseline_scores.csv
```

## Reciprocal Rank Fusion / Ensemble Weights

Your current hybrid retriever is built in `retriever.py`:

```python
def get_hybrid_retriever(keywords_weight=0.5, semantic_weight=0.5):
```

Try these pairs:

```text
BM25-heavy:      keywords_weight=0.7 semantic_weight=0.3
Balanced:        keywords_weight=0.5 semantic_weight=0.5
Semantic-heavy:  keywords_weight=0.3 semantic_weight=0.7
```

At the moment, `rag_chain.get_retriever()` calls `get_hybrid_retriever()` without exposing these values. To test weights cleanly, add config fields such as:

```python
hybrid_keywords_weight: float = 0.5
hybrid_semantic_weight: float = 0.5
```

Then pass them into `get_hybrid_retriever(...)`.

For every weight setting, run RAGAS and compare:

- `context_precision`: did ranking improve?
- `context_recall`: did retrieval miss fewer needed facts?
- `faithfulness`: did better retrieval reduce hallucination?

## Chunk Size and Overlap

Chunking is controlled in `config.py`:

```python
chunk_size: int = 1000
chunk_overlap: int = 200
```

Important: changing chunk size or overlap requires rebuilding Chroma, because existing chunks are already stored.

Suggested experiments:

```text
Small chunks:   chunk_size=500,  chunk_overlap=100
Baseline:       chunk_size=1000, chunk_overlap=200
Large chunks:   chunk_size=1500, chunk_overlap=300
```

Workflow:

1. Stop the app.
2. Change `chunk_size` and `chunk_overlap`.
3. Rebuild the vector database.
4. Run `eval/ragas_evaluate.py`.
5. Save outputs with descriptive names.

Example output names:

```bash
venv/bin/python eval/ragas_evaluate.py \
  --responses-out eval/chunk_500_100_responses.csv \
  --scores-out eval/chunk_500_100_scores.csv
```

## Semantic Chunking

Your current ingestion uses `RecursiveCharacterTextSplitter` in `ingest.py`. Semantic chunking would require replacing or adding another splitter.

A practical route:

1. Add a new config option:

```python
chunking_strategy: str = "recursive"
```

2. Keep the current recursive splitter as the baseline.
3. Add a semantic splitter path, for example LangChain's semantic chunker if installed.
4. Re-ingest documents into a separate Chroma collection name for each strategy.

Use separate collection names to avoid overwriting results:

```text
papers_recursive_1000_200
papers_recursive_500_100
papers_semantic_default
```

Then run the same RAGAS dataset against each collection and compare scores.

## What Good Results Look Like

As a rough starting point:

```text
faithfulness       >= 0.85
answer_relevancy   >= 0.80
context_precision  >= 0.70
context_recall     >= 0.70
```

Do not treat these as universal truth. Use them to compare versions of your own system.

## Most Important Rule

Only compare runs that use the same:

- dataset
- generator model
- evaluator model
- RAGAS metrics

Then change one retrieval/chunking parameter at a time.
