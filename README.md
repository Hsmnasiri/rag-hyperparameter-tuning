# RAG Hyperparameter Tuning

A comparative study of foundational search algorithms for RAG hyperparameter tuning.

This project implements and compares the performance of different metaheuristic algorithms for tuning the `chunk_size` and `top_k` hyperparameters of a Retrieval-Augmented Generation (RAG) system.  
The current implementation automatically downloads the full SQuAD v2.0 dev set (if it is not already cached under `data/`) and evaluates configurations on 500 answerable QA pairs by default, while building the knowledge base from all loaded contexts.

## Project Structure

- `data/`: Contains the dataset for the RAG pipeline.
- `notebooks/`: Jupyter notebooks for analysis and visualization.
- `src/`: Source code.
  - `rag/`: RAG pipeline and evaluator.
  - `algorithms/`: Search algorithms.
  - `main.py`: Main script to run the experiments.
- `tests/`: Unit tests.
- `pyproject.toml`: Project metadata and dependencies.

## Implementation Highlights

- **RAG pipeline:** `src/rag/pipeline.py` exposes a modular pipeline with either TF‑IDF or dense (SentenceTransformer) retrieval plus a configurable generation stage. The default generator is a seq2seq LLM (`google/flan-t5-base`) that conditions on the retrieved context; a lightweight extractive generator is still available for quick smoke tests.
- **Evaluator:** Automatically downloads SQuAD dev-v2.0 if needed, loads up to `RAG_DATASET_SIZE` QA pairs (default 500) for building the knowledge base, and uses a smaller evaluation slice (`RAG_EVAL_SAMPLE_SIZE`, default 80) for fast search iterations. Full-dataset evaluations are produced when exporting detailed predictions.
- **Search algorithms:** Random Search, Hill Climbing, and Simulated Annealing all interact with the shared evaluator, so they benefit from the same retriever/generator choices and dataset budget.
- **Experiment runner:** `src/main.py` executes repeated trials of each algorithm, logs per-run metrics to `results/experiment_results.csv`, exports per-question predictions (`results/<algorithm>_predictions.json`), and renders publication-ready plots in `results/plots/`.

## Installation

### Poetry (recommended)
```bash
poetry install
```

### Pip / virtualenv via `requirements.txt`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Run the experiments:
   ```bash
   poetry run python src/main.py
   ```
   If you activated a virtual environment via `requirements.txt`, run `python src/main.py` instead.

Environment toggles:
- `RAG_DATASET_SIZE=500` to enlarge the evaluation set.
- `RAG_EVAL_SAMPLE_SIZE=150` to change how many QA pairs each fitness evaluation uses.
- `RAG_RETRIEVER=tfidf` to fall back to the lightweight baseline (default is `dense`).
- `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2` to try a different encoder.
- `RAG_GENERATOR=seq2seq`/`extractive` and `RAG_GENERATOR_MODEL=google/flan-t5-base` (default is the lighter `flan-t5-small`) to control the answer generator.

After each run you’ll find:
- `results/experiment_results.csv`: raw per-run metrics.
- `results/summary.json`: best configuration per algorithm plus metadata.
- `results/*.json`: per-question predictions for each algorithm’s best run.
- `results/plots/*.png`: box plot + mean±std bar chart ready for presentations.
