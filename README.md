# RAG Hyperparameter Tuning

A comparative study of foundational search algorithms for RAG hyperparameter tuning.

This project implements and compares the performance of different metaheuristic algorithms for tuning the `chunk_size` and `top_k` hyperparameters of a Retrieval-Augmented Generation (RAG) system.

## 🎯 Research Questions

- **RQ1:** How do the search algorithms compare in their ability to find a high-performing RAG configuration (in terms of F1 score)?
- **RQ2:** What is the difference in computational cost (measured in number of evaluations) for each algorithm to converge on a "good enough" solution?

## 📁 Project Structure

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

## 🚀 Quick Start

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

### Launch Web Dashboard 🌐

```bash
uvicorn src.web.app:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## 🔬 Implemented Algorithms

| Algorithm | Description | Type |
|-----------|-------------|------|
| **Random Search** | Uniform random sampling of configurations | Baseline |
| **Hill Climbing** | Greedy local search with single-step neighbors | Local Search |
| **Simulated Annealing** | Probabilistic local search with cooling schedule | Metaheuristic |
| **Genetic Algorithm** | Population-based evolutionary optimization | Metaheuristic |

## 📊 Search Space

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `chunk_size` | 128 - 1024 | 64 | Size of text chunks for retrieval |
| `top_k` | 1 - 10 | 1 | Number of chunks to retrieve |

**Total configurations:** 15 × 10 = 150

## ⚙️ Configuration

Environment variables for customization:

```bash
# Dataset
RAG_DATASET_SIZE=500          # Number of QA pairs to load
RAG_EVAL_SAMPLE_SIZE=80       # QA pairs per evaluation

# Retriever
RAG_RETRIEVER=dense           # dense | tfidf
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Generator
RAG_GENERATOR=seq2seq         # seq2seq | extractive
RAG_GENERATOR_MODEL=google/flan-t5-small
RAG_GENERATOR_MAX_NEW_TOKENS=64
RAG_GENERATOR_TEMPERATURE=0.0
```

## 📈 Results

After running experiments, you'll find:

- `results/experiment_results.csv` - Raw per-run metrics
- `results/summary.json` - Best configuration per algorithm
- `results/plots/` - Visualization charts:
  - `best_score_boxplot.png` - Score distribution
  - `best_score_summary.png` - Mean ± Std comparison

## 🖥️ Web Dashboard Features

The interactive dashboard provides:

- **Overview Tab:** Real-time statistics and algorithm comparison charts
- **Run Experiment Tab:** Configure and execute experiments with live progress
- **Analysis Tab:** Deep dive into results with research question analysis
- **Configuration Tab:** View and understand search space parameters

## 📚 References

- Barker, M., et al. (2025). Faster, cheaper, better: Multi-objective hyperparameter optimization for LLM and RAG systems. arXiv:2502.18635
- Bulhakov, V., et al. (2025). Investigating the role of LLMs hyperparameter tuning and prompt engineering. arXiv:2507.14735
- Kim, J., et al. (2024). AutoRAG-HP: Automatic online hyper-parameter tuning for RAG. EMNLP 2024

## 👥 Authors

- Taha El Mouatadir
- Hesam Nasiri

## 📄 License

MIT License
