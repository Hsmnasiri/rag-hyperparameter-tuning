from __future__ import annotations

import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.random_search import random_search
from src.algorithms.simulated_annealing import simulated_annealing
from src.rag.evaluator import evaluate_rag_pipeline, evaluate_rag_with_details


# Define the search space
SEARCH_SPACE = {
    "chunk_size": list(range(128, 1025, 64)),
    "top_k": list(range(1, 11)),
}

# Define the number of evaluations and runs
MAX_EVALUATIONS = 20
NUM_RUNS = 10

# Experiment output locations
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Shared evaluator configuration so both algorithms and reporting stay in sync
RETRIEVER_TYPE = os.environ.get("RAG_RETRIEVER", "dense")
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EVALUATOR = partial(
    evaluate_rag_pipeline,
    retriever_type=RETRIEVER_TYPE,
    embedding_model=EMBEDDING_MODEL,
)

ALGORITHMS = {
    "Random Search": random_search,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
}


def run_experiment(
    algorithm_name: str, algorithm: Callable[[Dict[str, List[int]], int], tuple]
) -> List[Dict[str, object]]:
    """
    Runs `NUM_RUNS` executions of the provided algorithm and captures the best score/config.
    """
    records: List[Dict[str, object]] = []
    for run_idx in range(1, NUM_RUNS + 1):
        random.seed(run_idx)  # deterministic randomness per run
        best_config, best_score = algorithm(
            SEARCH_SPACE,
            MAX_EVALUATIONS,
            evaluator=EVALUATOR,
        )
        records.append(
            {
                "algorithm": algorithm_name,
                "run": run_idx,
                "best_score": best_score,
                "chunk_size": best_config["chunk_size"],
                "top_k": best_config["top_k"],
            }
        )
        print(
            f"[{algorithm_name}] Run {run_idx}/{NUM_RUNS} "
            f"=> best_score={best_score:.4f} "
            f"(chunk_size={best_config['chunk_size']}, top_k={best_config['top_k']})"
        )
    return records


def save_results_dataframe(records: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    output_path = RESULTS_DIR / "experiment_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved raw results to {output_path}")
    return df


def plot_distributions(df: pd.DataFrame) -> None:
    algorithms = list(df["algorithm"].unique())
    plt.style.use("seaborn-v0_8-colorblind")

    # Box plot of per-run best scores
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[df["algorithm"] == alg]["best_score"] for alg in algorithms]
    ax.boxplot(
        data,
        labels=algorithms,
        patch_artist=True,
        boxprops=dict(facecolor="#4C78A8", alpha=0.6),
        medianprops=dict(color="#F58518", linewidth=2),
    )
    ax.set_title("Per-Run Best F1 by Algorithm")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    boxplot_path = PLOTS_DIR / "best_score_boxplot.png"
    fig.savefig(boxplot_path, dpi=300)
    plt.close(fig)
    print(f"Saved box plot to {boxplot_path}")

    # Mean ± std bar chart
    summary = df.groupby("algorithm")["best_score"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary["algorithm"],
        summary["mean"],
        yerr=summary["std"],
        capsize=6,
        color=["#4C78A8", "#F58518", "#54A24B", "#EECA3B"][: len(summary)],
        alpha=0.85,
    )
    ax.set_ylabel("Mean F1 Score")
    ax.set_ylim(0, 1)
    ax.set_title("Mean ± Std Dev of Best F1 Scores")
    for bar, mean in zip(bars, summary["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    barplot_path = PLOTS_DIR / "best_score_summary.png"
    fig.savefig(barplot_path, dpi=300)
    plt.close(fig)
    print(f"Saved summary bar chart to {barplot_path}")


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_")


def export_predictions(df: pd.DataFrame) -> None:
    summary = {}
    for algorithm_name, group in df.groupby("algorithm"):
        best_row = group.sort_values("best_score", ascending=False).iloc[0]
        mean_score, details = evaluate_rag_with_details(
            best_row["chunk_size"],
            best_row["top_k"],
            retriever_type=RETRIEVER_TYPE,
            embedding_model=EMBEDDING_MODEL,
        )
        summary[algorithm_name] = {
            "best_run": int(best_row["run"]),
            "chunk_size": int(best_row["chunk_size"]),
            "top_k": int(best_row["top_k"]),
            "best_score": float(best_row["best_score"]),
            "recomputed_score": float(mean_score),
            "num_questions": len(details),
        }
        output_path = RESULTS_DIR / f"{slugify(algorithm_name)}_predictions.json"
        with output_path.open("w") as f:
            json.dump(details, f, indent=2)
        print(f"Stored per-question predictions for {algorithm_name} at {output_path}")

    summary_path = RESULTS_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metadata to {summary_path}")


def main():
    print("Starting RAG hyperparameter search experiments...")
    start_time = time.time()
    all_records: List[Dict[str, object]] = []
    for algorithm_name, algorithm_fn in ALGORITHMS.items():
        records = run_experiment(algorithm_name, algorithm_fn)
        all_records.extend(records)

    df = save_results_dataframe(all_records)
    plot_distributions(df)
    export_predictions(df)
    elapsed = time.time() - start_time
    minutes, secs = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)
    formatted = (
        f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s" if minutes else f"{secs}s"
    )
    print(f"\nExperiments complete. Elapsed: {formatted}.")


if __name__ == "__main__":
    main()
