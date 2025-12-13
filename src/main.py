"""
Main experiment runner for RAG hyperparameter tuning.

Features:
- Parallel algorithm runs
- Full quality evaluation (500 dataset, 100 eval sample)
- Comprehensive statistics and visualization
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.random_search import random_search
from src.algorithms.simulated_annealing import simulated_annealing
from src.rag.evaluator import evaluate_rag_pipeline, get_evaluator
from src.rag.search_space import DEFAULT_SEARCH_SPACE


# ============================================================================
# SEARCH SPACE - Full range
# ============================================================================
SEARCH_SPACE = DEFAULT_SEARCH_SPACE.get_config_space()
TOTAL_CONFIGS = DEFAULT_SEARCH_SPACE.get_total_configurations()

# ============================================================================
# EXPERIMENT PARAMETERS - Full quality
# ============================================================================
MAX_EVALUATIONS = 50  # Literature-aligned budget (~50 evaluations)
NUM_RUNS = 10         # Statistical significance

# Output directories
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Algorithm registry
ALGORITHMS = {
    "Random Search": random_search,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
}


def run_single_trial(
    algorithm_name: str,
    algorithm_fn: Callable,
    run_idx: int,
    max_evaluations: int,
) -> Dict:
    """Run a single trial of an algorithm."""
    # Set deterministic seed
    random.seed(run_idx * 42 + hash(algorithm_name) % 1000)
    np.random.seed(run_idx * 42 + hash(algorithm_name) % 1000)
    
    start_time = time.time()
    best_config, best_score = algorithm_fn(
        SEARCH_SPACE,
        max_evaluations,
        evaluator=evaluate_rag_pipeline,
    )
    elapsed = time.time() - start_time
    
    return {
        "algorithm": algorithm_name,
        "run": run_idx,
        "best_score": best_score,
        "chunk_size": best_config.get("chunk_size"),
        "chunk_overlap": best_config.get("chunk_overlap"),
        "top_k": best_config.get("top_k"),
        "similarity_threshold": best_config.get("similarity_threshold"),
        "retrieval_metric": best_config.get("retrieval_metric"),
        "embedding_model": best_config.get("embedding_model"),
        "max_context_chars": best_config.get("max_context_chars"),
        "time_seconds": elapsed,
    }


def run_experiment_sequential(
    algorithm_name: str, 
    algorithm_fn: Callable,
    num_runs: int = NUM_RUNS,
    max_evaluations: int = MAX_EVALUATIONS,
) -> List[Dict]:
    """Run multiple trials of an algorithm sequentially."""
    records = []
    
    for run_idx in range(1, num_runs + 1):
        result = run_single_trial(algorithm_name, algorithm_fn, run_idx, max_evaluations)
        records.append(result)

        config_str = (
            f"chunk={result['chunk_size']}, overlap={result['chunk_overlap']}, "
            f"top_k={result['top_k']}, thr={result['similarity_threshold']}, "
            f"metric={result['retrieval_metric']}, embed={result['embedding_model']}"
        )
        print(
            f"[{algorithm_name}] Run {run_idx}/{num_runs} "
            f"=> score={result['best_score']:.4f} "
            f"({config_str}) "
            f"[{result['time_seconds']:.1f}s]"
        )
    
    return records


def run_all_experiments_parallel(
    num_workers: int = 3,
    num_runs: int = NUM_RUNS,
    max_evaluations: int = MAX_EVALUATIONS,
) -> List[Dict]:
    """
    Run all algorithm experiments with parallelization.
    
    Note: We parallelize across algorithms, not within,
    because the evaluator has pre-computed caches.
    """
    all_records = []
    
    # Run algorithms in parallel using threads
    # (ProcessPoolExecutor would require serializing the evaluator)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        for algorithm_name, algorithm_fn in ALGORITHMS.items():
            for run_idx in range(1, num_runs + 1):
                future = executor.submit(
                    run_single_trial,
                    algorithm_name,
                    algorithm_fn,
                    run_idx,
                    max_evaluations,
                )
                futures[future] = (algorithm_name, run_idx)
        
        completed = 0
        total = len(futures)
        
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            algorithm_name, run_idx = futures[future]
            result = future.result()
            all_records.append(result)
            completed += 1

            config_str = (
                f"chunk={result['chunk_size']}, overlap={result['chunk_overlap']}, "
                f"top_k={result['top_k']}, thr={result['similarity_threshold']}, "
                f"metric={result['retrieval_metric']}, embed={result['embedding_model']}"
            )
            print(
                f"[{completed}/{total}] [{algorithm_name}] Run {run_idx} "
                f"=> score={result['best_score']:.4f} "
                f"({config_str}) "
                f"[{result['time_seconds']:.1f}s]"
            )
    
    return all_records


def save_results(records: List[Dict]) -> pd.DataFrame:
    """Save results to CSV."""
    df = pd.DataFrame(records)
    df = df.sort_values(["algorithm", "run"]).reset_index(drop=True)
    output_path = RESULTS_DIR / "experiment_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    return df


def plot_results(df: pd.DataFrame) -> None:
    """Generate visualization plots."""
    algorithms = list(ALGORITHMS.keys())
    colors = ["#6366f1", "#10b981", "#f59e0b"]
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[df["algorithm"] == alg]["best_score"].values for alg in algorithms]
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score Distribution by Algorithm (Full Quality)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "best_score_boxplot.png", dpi=300)
    plt.close(fig)
    print(f"Saved box plot to {PLOTS_DIR / 'best_score_boxplot.png'}")
    
    # 2. Mean ± Std bar chart
    summary = df.groupby("algorithm")["best_score"].agg(["mean", "std"]).reset_index()
    summary = summary.set_index("algorithm").loc[algorithms].reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        summary["algorithm"],
        summary["mean"],
        yerr=summary["std"],
        capsize=8,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Mean ± Std of Best Scores", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    
    for bar, mean in zip(bars, summary["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.03,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "best_score_summary.png", dpi=300)
    plt.close(fig)
    print(f"Saved summary bar chart to {PLOTS_DIR / 'best_score_summary.png'}")
    
    # 3. Run-by-run line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for alg, color in zip(algorithms, colors):
        alg_data = df[df["algorithm"] == alg].sort_values("run")
        ax.plot(
            alg_data["run"],
            alg_data["best_score"],
            marker="o",
            label=alg,
            color=color,
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score by Run", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, NUM_RUNS + 1))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_by_run.png", dpi=300)
    plt.close(fig)
    print(f"Saved run plot to {PLOTS_DIR / 'score_by_run.png'}")
    
    # 4. Heatmap of chunk_size vs top_k
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (alg, color) in zip(axes, zip(algorithms, colors)):
        alg_data = df[df["algorithm"] == alg]
        
        # Create a simple scatter plot
        scatter = ax.scatter(
            alg_data["chunk_size"],
            alg_data["top_k"],
            c=alg_data["best_score"],
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="white",
        )
        ax.set_xlabel("Chunk Size")
        ax.set_ylabel("Top K")
        ax.set_title(alg)
        plt.colorbar(scatter, ax=ax, label="Score")
    
    fig.suptitle("Best Configurations Found", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "configuration_scatter.png", dpi=300)
    plt.close(fig)
    print(f"Saved configuration scatter to {PLOTS_DIR / 'configuration_scatter.png'}")


def export_summary(df: pd.DataFrame) -> None:
    """Export summary statistics."""
    summary = {}
    
    for algorithm in ALGORITHMS.keys():
        alg_data = df[df["algorithm"] == algorithm]
        best_row = alg_data.loc[alg_data["best_score"].idxmax()]
        
        summary[algorithm] = {
            "mean_score": float(alg_data["best_score"].mean()),
            "std_score": float(alg_data["best_score"].std()),
            "max_score": float(alg_data["best_score"].max()),
            "min_score": float(alg_data["best_score"].min()),
            "best_chunk_size": int(best_row["chunk_size"]),
            "best_chunk_overlap": int(best_row.get("chunk_overlap", 0)),
            "best_top_k": int(best_row["top_k"]),
            "best_similarity_threshold": float(best_row.get("similarity_threshold", 0.0)),
            "best_retrieval_metric": str(best_row.get("retrieval_metric", "")),
            "best_embedding_model": str(best_row.get("embedding_model", "")),
            "best_max_context_chars": int(best_row.get("max_context_chars", 0)),
            "best_run": int(best_row["run"]),
            "avg_time_seconds": float(alg_data["time_seconds"].mean()),
            "total_runs": len(alg_data),
        }
    
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Mean':>10} {'Std':>10} {'Best':>10} {'Config':>28} {'Avg Time':>10}")
    print("-" * 80)
    for alg, stats in summary.items():
        config = (
            f"cs={stats['best_chunk_size']},ov={stats['best_chunk_overlap']},"
            f"k={stats['best_top_k']},thr={stats['best_similarity_threshold']},"
            f"m={stats['best_retrieval_metric']},"
            f"emb={stats['best_embedding_model']},"
            f"ctx={stats['best_max_context_chars']}"
        )
        print(
            f"{alg:<25} {stats['mean_score']:>10.4f} {stats['std_score']:>10.4f} "
            f"{stats['max_score']:>10.4f} {config:>28} {stats['avg_time_seconds']:>9.1f}s"
        )
    print("=" * 80)
    
    # Statistical comparison
    print("\n📊 Statistical Analysis:")
    algorithms = list(ALGORITHMS.keys())
    for i, alg1 in enumerate(algorithms):
        for alg2 in algorithms[i+1:]:
            scores1 = df[df["algorithm"] == alg1]["best_score"].values
            scores2 = df[df["algorithm"] == alg2]["best_score"].values
            
            # Simple t-test
            from scipy import stats as scipy_stats
            try:
                t_stat, p_value = scipy_stats.ttest_ind(scores1, scores2)
                significance = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
                print(f"  {alg1} vs {alg2}: p={p_value:.4f} ({significance})")
            except:
                print(f"  {alg1} vs {alg2}: (scipy not available for statistical test)")


def main():
    """Main entry point."""
    print("=" * 80)
    print("RAG HYPERPARAMETER TUNING EXPERIMENT")
    print("High-Performance Parallel Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Search space size: {TOTAL_CONFIGS} configurations")
    print(f"  - chunk_size=[{min(SEARCH_SPACE['chunk_size'])}..{max(SEARCH_SPACE['chunk_size'])}], "
          f"overlap=[{min(SEARCH_SPACE['chunk_overlap'])}..{max(SEARCH_SPACE['chunk_overlap'])}], "
          f"top_k=[{min(SEARCH_SPACE['top_k'])}..{max(SEARCH_SPACE['top_k'])}]")
    print(f"  - similarity_thresholds={SEARCH_SPACE['similarity_threshold']} | metrics={SEARCH_SPACE['retrieval_metric']}")
    print(f"  - embedding_models={SEARCH_SPACE['embedding_model']} | context_windows={SEARCH_SPACE['max_context_chars']}")
    print(f"  - Max evaluations per run: {MAX_EVALUATIONS}")
    print(f"  - Number of runs: {NUM_RUNS}")
    print(f"  - Algorithms: {', '.join(ALGORITHMS.keys())}")
    print(f"  - Quality: Full (500 dataset, 100 eval sample)")
    print("=" * 80)
    
    # Initialize evaluator (pre-computes embeddings)
    print("\n🚀 Initializing high-performance evaluator...")
    start_time = time.time()
    evaluator = get_evaluator()
    init_time = time.time() - start_time
    print(f"✓ Initialization complete in {init_time:.1f}s")
    print(f"  • Dataset size: {len(evaluator.dataset)}")
    print(f"  • Eval sample: {len(evaluator.eval_dataset)}")
    print(f"  • Documents: {len(evaluator.documents)}")
    print(f"  • Parallel workers: {evaluator.num_workers}")
    
    # Run experiments
    print("\n" + "=" * 80)
    print("Running experiments...")
    print("=" * 80)
    
    experiment_start = time.time()
    
    # Sequential mode (more reliable, still fast due to caching)
    all_records = []
    for algorithm_name, algorithm_fn in ALGORITHMS.items():
        print(f"\n{'='*50}")
        print(f"Running {algorithm_name}...")
        print(f"{'='*50}")
        records = run_experiment_sequential(algorithm_name, algorithm_fn)
        all_records.extend(records)
    
    total_time = time.time() - experiment_start
    
    # Save and visualize
    df = save_results(all_records)
    plot_results(df)
    export_summary(df)
    
    # Final timing
    minutes, secs = divmod(int(total_time), 60)
    hours, minutes = divmod(minutes, 60)
    time_str = f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"
    print(f"\n⏱️ Total experiment time: {time_str}")
    print(f"📁 Results saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
