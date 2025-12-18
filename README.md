# RAG Hyperparameter Tuning

A comparative study of foundational search algorithms for RAG hyperparameter tuning.

This project implements and compares different metaheuristic algorithms for tuning a Retrieval-Augmented Generation (RAG) pipeline across chunking and retrieval hyperparameters (and a small set of retrieval-model choices).

## Research Questions

- **RQ1:** How do the search algorithms compare in their ability to find a high-performing RAG configuration (in terms of F1 score)?
- **RQ2:** What is the difference in computational cost (measured in number of evaluations) for each algorithm to converge on a "good enough" solution?

## Project Structure

```
rag-hyperparameter-tuning/
├── data/                    # Dataset files (SQuAD v2.0)
├── results/                 # Experiment outputs
│   ├── plots/              # Generated visualizations
│   ├── experiment_results.csv
│   └── summary.json
├── src/
│   ├── algorithms/         # Search algorithms
│   │   ├── random_search.py
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   └── genetic_algorithm.py
│   ├── rag/               # RAG pipeline
│   │   ├── pipeline.py
│   │   └── evaluator.py
│   ├── web/               # Web dashboard
│   │   └── app.py
│   └── main.py            # CLI experiment runner
├── tests/                  # Unit tests
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### Installation

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Run Experiments (CLI)

```bash
python -m src.main
```

## Run Recipes (All Modes)

All commands below assume you have the venv activated or you call `.venv/bin/python`.

### 1) Full RAG tuning (default)

Runs all three algorithms sequentially, with live progress under `results/live/`.

```bash
.venv/bin/python -m src.main
```

### 2) Run each algorithm as a separate command (recommended for multi-machine)

```bash
# Random Search
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/random_search RAG_SKIP_PLOTS=1 RAG_SKIP_SUMMARY=1 .venv/bin/python -m src.main

# Hill Climbing
RAG_ALGORITHM=hill_climbing RAG_RESULTS_DIR=results/hill_climbing RAG_SKIP_PLOTS=1 RAG_SKIP_SUMMARY=1 .venv/bin/python -m src.main

# Simulated Annealing
RAG_ALGORITHM=simulated_annealing RAG_RESULTS_DIR=results/simulated_annealing RAG_SKIP_PLOTS=1 RAG_SKIP_SUMMARY=1 .venv/bin/python -m src.main
```

Optional: split one algorithm’s runs across machines:

```bash
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/rs_a RAG_RUN_START=1 RAG_RUN_END=5  .venv/bin/python -m src.main
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/rs_b RAG_RUN_START=6 RAG_RUN_END=10 .venv/bin/python -m src.main
```

### 3) Baseline-only (No-RAG LLM)

This runs the **LLM-only** baseline (no retrieval) and exits.

Small baseline (fast, noisy):

```bash
RAG_BASELINE_ONLY=1 RAG_TRACE_BASELINE=1 \
RAG_DATASET_SIZE=1000 RAG_EVAL_SAMPLE_SIZE=100 \
RAG_RESULTS_DIR=results/baseline_100 .venv/bin/python -m src.main
```

Larger baseline (example used during development: 2000 questions):

```bash
RAG_BASELINE_ONLY=1 RAG_TRACE_BASELINE=1 \
RAG_DATASET_SIZE=5928 RAG_EVAL_SAMPLE_SIZE=2000 \
RAG_RESULTS_DIR=results/baseline_2000 .venv/bin/python -m src.main
```

### 4) Enable per-question tracing (RAG)

Writes per-question logs (retrieved contexts + prompt + prediction + metrics). This can be large.

```bash
RAG_TRACE_QA=1 RAG_RESULTS_DIR=results/random_search \
RAG_ALGORITHM=random_search .venv/bin/python -m src.main
```

### 5) Merge distributed results and generate plots (one machine)

After you run the algorithms on multiple machines and copy/push the result folders into one workspace, run:

```bash
.venv/bin/python -m src.report results/random_search results/hill_climbing results/simulated_annealing --out results/combined
```

This produces:
- `results/combined/experiment_results.csv`
- `results/combined/summary.json`
- `results/combined/plots/*.png`

### Run Algorithms Separately (3 Commands)

Useful when you want to run each algorithm on a different machine (or in parallel terminal sessions).

```bash
# Random Search
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/random_search python -m src.main

# Hill Climbing
RAG_ALGORITHM=hill_climbing RAG_RESULTS_DIR=results/hill_climbing python -m src.main

# Simulated Annealing
RAG_ALGORITHM=simulated_annealing RAG_RESULTS_DIR=results/simulated_annealing python -m src.main
```

Optional: split the same algorithm’s runs across machines:

```bash
# Machine A runs 1..5
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/rs_a RAG_RUN_START=1 RAG_RUN_END=5 python -m src.main

# Machine B runs 6..10
RAG_ALGORITHM=random_search RAG_RESULTS_DIR=results/rs_b RAG_RUN_START=6 RAG_RUN_END=10 python -m src.main
```

### Launch Web Dashboard

```bash
uvicorn src.web.app:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Implemented Algorithms

| Algorithm | Description | Type |
|-----------|-------------|------|
| **Random Search** | Uniform random sampling of configurations | Baseline |
| **Hill Climbing** | Greedy local search with single-step neighbors | Local Search |
| **Simulated Annealing** | Probabilistic local search with cooling schedule | Metaheuristic |
| **Genetic Algorithm** | Population-based evolutionary optimization | Metaheuristic |

## Search Space

The search space is defined in `src/rag/search_space.py` and is discrete:

| Parameter | Values | Count | Description |
|-----------|--------|-------|-------------|
| `chunk_size` | `[128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024]` | 11 | Chunk size (characters) |
| `chunk_overlap` | `[0, 16, 32, 48, 64, 80]` | 6 | Overlap between chunks |
| `top_k` | `1..10` | 10 | Number of chunks to retrieve |
| `similarity_threshold` | `[0.2, 0.4, 0.6]` | 3 | Minimum retrieval similarity |
| `embedding_model` | `[minilm, mpnet, bge]` | 3 | Dense embedding model key |
| `retrieval_metric` | `[cosine, dot]` | 2 | Similarity metric for dense retrieval |
| `max_context_chars` | `[1024, 2048, 3072]` | 3 | Context budget passed to generator |

**Total configurations:** 11 × 6 × 10 × 3 × 3 × 2 × 3 = **35,640**

## Configuration

Environment variables for customization (defaults shown):

```bash
# Dataset + sampling
RAG_DATASET_PATH=data/squad_dev_v2.0.json
RAG_DATASET_SIZE=500          # Number of answerable QA pairs to load
RAG_EVAL_SAMPLE_SIZE=100      # QA pairs per evaluation
RAG_DATASET_SEED=42           # Deterministic sampling when subsetting

# Experiment budget (CLI runner)
RAG_MAX_EVALUATIONS=50
RAG_NUM_RUNS=10
RAG_RUN_START=1
RAG_RUN_END=10
RAG_ALGORITHM=all             # all | random_search | hill_climbing | simulated_annealing
RAG_RESULTS_DIR=results       # Output folder (use separate dirs to parallelize)
RAG_BASELINE_ONLY=0           # 1 = run No-RAG LLM baseline then exit
RAG_SKIP_PLOTS=0              # 1 = skip plot generation (useful for distributed runs)
RAG_SKIP_SUMMARY=0            # 1 = skip summary.json generation (useful for distributed runs)

# Pipeline toggles
RAG_RETRIEVER=dense           # dense | tfidf (tfidf is a fast baseline)
RAG_GENERATOR=seq2seq         # seq2seq | extractive
RAG_ENABLE_SEMANTIC_SIM=1     # 1 | 0 (disable for faster smoke tests)

# Generator (seq2seq only)
RAG_GENERATOR_MODEL=flan-t5-small   # or google/flan-t5-small
RAG_GENERATOR_MAX_NEW_TOKENS=64
RAG_GENERATOR_TEMPERATURE=0.0

# Tracing (optional; can be large)
RAG_TRACE_QA=0
RAG_TRACE_BASELINE=0
RAG_TRACE_MAX_CONTEXTS=5
RAG_TRACE_MAX_CONTEXT_CHARS=300
RAG_TRACE_MAX_PROMPT_CHARS=2000
RAG_TRACE_MAX_PREDICTION_CHARS=500
```

### Why not use the full dataset by default?

Hyperparameter tuning evaluates **many** candidate configurations (e.g., `RAG_MAX_EVALUATIONS × RAG_NUM_RUNS × algorithms`). Each configuration evaluation runs the full RAG pipeline over `RAG_EVAL_SAMPLE_SIZE` questions, including retrieval and (optionally) LLM generation, so runtime scales **roughly linearly** with `RAG_EVAL_SAMPLE_SIZE`.

`RAG_DATASET_SIZE` exists because in RAG the dataset also defines the **retrieval corpus** (unique contexts → chunks → embeddings). Using the full corpus for every run can be very slow and memory-heavy, especially when you sweep over different `chunk_size`, `chunk_overlap`, and embedding models.

Recommended workflow:
- Tune on a reasonable subset (e.g., `RAG_DATASET_SIZE=500`, `RAG_EVAL_SAMPLE_SIZE=100`) to iterate quickly and compare algorithms fairly.
- Re-evaluate the best found configurations on the full answerable set for the final report.

## Live Progress Files

During runs, the CLI writes a few files under `results/live/` (or under `RAG_RESULTS_DIR/live/` if you override it):

- `progress.json` (overwritten frequently): heartbeat/status + per-question progress inside an evaluation
- `evaluations.jsonl` (append-only): one line per completed configuration evaluation (score + config + elapsed time)
- `runs.jsonl` (append-only): one line per completed run (best config + score)
- `qa_traces.jsonl` (optional, can be large): per-question logs with retrieved contexts + prompt + prediction (enable with `RAG_TRACE_QA=1`)
  - For the No-RAG baseline traces, enable `RAG_TRACE_BASELINE=1` (baseline prompt + prediction + metrics)

Note: The SQuAD v2.0 dev file contains 11,873 QAs (including unanswerable). This project evaluates only answerable items; the full answerable set is 5,928. To run on the full set, set `RAG_DATASET_SIZE=5928` and (optionally) `RAG_EVAL_SAMPLE_SIZE=5928`.

## Results

After running experiments, you'll find:

- `results/experiment_results.csv` - Raw per-run metrics
- `results/summary.json` - Best configuration per algorithm
- `results/plots/` - Visualization charts:
  - `best_score_boxplot.png` - Score distribution
  - `best_score_summary.png` - Mean ± Std comparison

## Web Dashboard Features

The interactive dashboard provides:

- **Overview Tab:** Real-time statistics and algorithm comparison charts
- **Run Experiment Tab:** Configure and execute experiments with live progress
- **Analysis Tab:** Deep dive into results with research question analysis
- **Configuration Tab:** View and understand search space parameters

## References

- Barker, M., et al. (2025). Faster, cheaper, better: Multi-objective hyperparameter optimization for LLM and RAG systems. arXiv:2502.18635
- Bulhakov, V., et al. (2025). Investigating the role of LLMs hyperparameter tuning and prompt engineering. arXiv:2507.14735
- Kim, J., et al. (2024). AutoRAG-HP: Automatic online hyper-parameter tuning for RAG. EMNLP 2024

## Authors

- Taha El Mouatadir
- Hesam Nasiri

## License

MIT License
