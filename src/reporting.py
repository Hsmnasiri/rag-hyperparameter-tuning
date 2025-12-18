from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_COLUMNS = [
    "algorithm",
    "run",
    "best_score",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "similarity_threshold",
    "retrieval_metric",
    "embedding_model",
    "max_context_chars",
    "time_seconds",
]


def save_results(records: List[Dict], *, results_dir: Path) -> pd.DataFrame:
    """Save results to CSV and return the DataFrame."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["algorithm", "run"]).reset_index(drop=True)
    output_path = results_dir / "experiment_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    return df


def plot_results(df: pd.DataFrame, *, plots_dir: Path, algorithms: List[str]) -> None:
    """Generate visualization plots for the provided algorithms."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No records to plot.")
        return

    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#0ea5e9"]
    colors = colors[: max(1, len(algorithms))]

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[df["algorithm"] == alg]["best_score"].values for alg in algorithms]
    bp = ax.boxplot(data, tick_labels=algorithms, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score Distribution by Algorithm", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "best_score_boxplot.png", dpi=300)
    plt.close(fig)
    print(f"Saved box plot to {plots_dir / 'best_score_boxplot.png'}")

    # 2) Mean ± Std
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
            float(mean) + 0.03,
            f"{float(mean):.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(plots_dir / "best_score_summary.png", dpi=300)
    plt.close(fig)
    print(f"Saved summary bar chart to {plots_dir / 'best_score_summary.png'}")

    # 3) Run-by-run line plot
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
    run_ticks = sorted({int(x) for x in df["run"].tolist() if pd.notna(x)})
    if run_ticks:
        ax.set_xticks(run_ticks)
    fig.tight_layout()
    fig.savefig(plots_dir / "score_by_run.png", dpi=300)
    plt.close(fig)
    print(f"Saved run plot to {plots_dir / 'score_by_run.png'}")

    # 4) Scatter of chunk_size vs top_k
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5))
    if len(algorithms) == 1:
        axes = [axes]

    for ax, (alg, color) in zip(axes, zip(algorithms, colors)):
        alg_data = df[df["algorithm"] == alg]
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
    fig.savefig(plots_dir / "configuration_scatter.png", dpi=300)
    plt.close(fig)
    print(f"Saved configuration scatter to {plots_dir / 'configuration_scatter.png'}")


def export_summary(
    df: pd.DataFrame,
    *,
    results_dir: Path,
    algorithms: List[str],
    baselines: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Export summary.json and print summary table; returns the summary dict."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}

    for algorithm in algorithms:
        alg_data = df[df["algorithm"] == algorithm]
        if alg_data.empty:
            continue

        best_row = alg_data.loc[alg_data["best_score"].idxmax()]
        std_score = float(alg_data["best_score"].std())
        if np.isnan(std_score):
            std_score = 0.0

        summary[algorithm] = {
            "mean_score": float(alg_data["best_score"].mean()),
            "std_score": std_score,
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
            "total_runs": int(len(alg_data)),
        }

    if baselines:
        summary["_baselines"] = baselines

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(
        f"{'Algorithm':<25} {'Mean':>10} {'Std':>10} {'Best':>10} {'Config':>28} {'Avg Time':>10}"
    )
    print("-" * 80)
    for alg in algorithms:
        if alg not in summary:
            continue
        stats = summary[alg]  # type: ignore[assignment]
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

    if baselines and "no_rag_llm_mean_fitness" in baselines:
        try:
            baseline_score = float(baselines["no_rag_llm_mean_fitness"])
        except (TypeError, ValueError):
            baseline_score = None

        if baseline_score is not None:
            print("\n" + "=" * 80)
            print("BASELINE COMPARISON")
            print("=" * 80)
            print(f"No-RAG LLM mean fitness: {baseline_score:.4f}")
            for alg in algorithms:
                if alg not in summary:
                    continue
                stats = summary[alg]  # type: ignore[assignment]
                delta = float(stats["max_score"]) - baseline_score
                print(f"{alg}: best={stats['max_score']:.4f} (delta={delta:+.4f})")
            print("=" * 80)

    print("\nStatistical Analysis:")
    for i, alg1 in enumerate(algorithms):
        for alg2 in algorithms[i + 1 :]:
            scores1 = df[df["algorithm"] == alg1]["best_score"].values
            scores2 = df[df["algorithm"] == alg2]["best_score"].values
            if len(scores1) < 2 or len(scores2) < 2:
                print(f"  {alg1} vs {alg2}: (not enough runs for statistical test)")
                continue
            try:
                from scipy import stats as scipy_stats

                _t_stat, p_value = scipy_stats.ttest_ind(scores1, scores2)
                significance = "Significant" if p_value < 0.05 else "Not significant"
                print(f"  {alg1} vs {alg2}: p={p_value:.4f} ({significance})")
            except Exception:
                print(f"  {alg1} vs {alg2}: (scipy not available for statistical test)")

    return summary

