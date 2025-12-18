"""
Main experiment runner for RAG hyperparameter tuning.

Features:
- Parallel algorithm runs
- Full quality evaluation (500 dataset, 100 eval sample)
- Comprehensive statistics and visualization
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.random_search import random_search
from src.algorithms.simulated_annealing import simulated_annealing
from src.rag.evaluator import evaluate_rag_pipeline, get_evaluator
from src.rag.search_space import DEFAULT_SEARCH_SPACE
from src.reporting import export_summary, plot_results, save_results


# ============================================================================
# SEARCH SPACE - Full range
# ============================================================================
SEARCH_SPACE = DEFAULT_SEARCH_SPACE.get_config_space()
TOTAL_CONFIGS = DEFAULT_SEARCH_SPACE.get_total_configurations()

# ============================================================================
# EXPERIMENT PARAMETERS - Full quality
# ============================================================================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


MAX_EVALUATIONS = max(1, _env_int("RAG_MAX_EVALUATIONS", 30))  # Literature-aligned budget (~50 evaluations)
NUM_RUNS = max(1, _env_int("RAG_NUM_RUNS", 10))               # Statistical significance

# Output directories
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", "results")).expanduser()
LIVE_DIR = RESULTS_DIR / "live"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
LIVE_DIR.mkdir(exist_ok=True, parents=True)

# Algorithm registry
ALGORITHMS = {
    "Random Search": random_search,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
}

_ALGORITHM_ALIASES = {
    "random": "Random Search",
    "random_search": "Random Search",
    "rs": "Random Search",
    "hill": "Hill Climbing",
    "hill_climbing": "Hill Climbing",
    "hc": "Hill Climbing",
    "simulated_annealing": "Simulated Annealing",
    "annealing": "Simulated Annealing",
    "sa": "Simulated Annealing",
}

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_algorithm_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _select_algorithms() -> Dict[str, Callable]:
    """
    Select algorithms to run based on `RAG_ALGORITHM`.

    Examples:
      - RAG_ALGORITHM=all (default)
      - RAG_ALGORITHM=random_search
      - RAG_ALGORITHM=hill_climbing,simulated_annealing
    """
    raw = os.environ.get("RAG_ALGORITHM", "all").strip()
    if raw == "" or _normalize_algorithm_token(raw) in {"all", "*"}:
        return dict(ALGORITHMS)

    selected: Dict[str, Callable] = {}
    for part in raw.split(","):
        token = _normalize_algorithm_token(part)
        if not token:
            continue
        display = _ALGORITHM_ALIASES.get(token)
        if display is None or display not in ALGORITHMS:
            valid = ", ".join(sorted(set(_ALGORITHM_ALIASES.keys()) | {"all"}))
            raise ValueError(f"Unknown RAG_ALGORITHM={part!r}. Valid values: {valid}")
        selected[display] = ALGORITHMS[display]

    if not selected:
        raise ValueError("RAG_ALGORITHM did not select any algorithm")
    return selected


@dataclass
class LiveLogger:
    live_dir: Path
    results_csv: Path
    lock: Lock = field(default_factory=Lock)

    @property
    def progress_path(self) -> Path:
        return self.live_dir / "progress.json"

    @property
    def evals_path(self) -> Path:
        return self.live_dir / "evaluations.jsonl"

    @property
    def runs_path(self) -> Path:
        return self.live_dir / "runs.jsonl"

    @property
    def qa_traces_path(self) -> Path:
        return self.live_dir / "qa_traces.jsonl"

    @property
    def qa_traces_dir(self) -> Path:
        return self.live_dir / "qa_traces"

    def reset(self, *, total_runs: int, max_evaluations: int) -> None:
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.evals_path.write_text("", encoding="utf-8")
        self.runs_path.write_text("", encoding="utf-8")
        self.qa_traces_path.write_text("", encoding="utf-8")
        # Clear per-evaluation trace files
        self.qa_traces_dir.mkdir(parents=True, exist_ok=True)
        for f in self.qa_traces_dir.glob("*.jsonl"):
            try:
                f.unlink()
            except Exception:
                pass
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
            writer.writeheader()
        self.update_progress(
            status="initialized",
            started_at=_utc_now_iso(),
            total_runs=total_runs,
            max_evaluations=max_evaluations,
            completed_runs=0,
        )

    def update_progress(self, **fields: object) -> None:
        payload = {"updated_at": _utc_now_iso(), **fields}
        with self.lock:
            self.progress_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def log_evaluation(self, record: Dict[str, object]) -> None:
        with self.lock:
            with self.evals_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_run(self, record: Dict[str, object]) -> None:
        with self.lock:
            with self.runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with self.results_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
                writer.writerow({k: record.get(k) for k in RUN_COLUMNS})


def run_single_trial(
    algorithm_name: str,
    algorithm_fn: Callable,
    run_idx: int,
    max_evaluations: int,
    live: LiveLogger,
    completed_runs: int,
    total_runs: int,
) -> Dict:
    """Run a single trial of an algorithm."""
    # Set deterministic seed
    random.seed(run_idx * 42 + hash(algorithm_name) % 1000)
    np.random.seed(run_idx * 42 + hash(algorithm_name) % 1000)

    eval_counter = 0

    def logged_evaluator(*args, **kwargs) -> float:
        nonlocal eval_counter
        eval_counter += 1

        config_data: Dict[str, object] = {}
        if len(args) == 1 and isinstance(args[0], dict):
            config_data.update(dict(args[0]))
        elif len(args) >= 2:
            config_data["chunk_size"] = args[0]
            config_data["top_k"] = args[1]

        config_dict_kw = kwargs.pop("config_dict", None)
        if isinstance(config_dict_kw, dict):
            config_data.update(dict(config_dict_kw))

        if "chunk_size" in kwargs:
            config_data["chunk_size"] = kwargs.pop("chunk_size")
        if "top_k" in kwargs:
            config_data["top_k"] = kwargs.pop("top_k")
        if kwargs:
            config_data.update(kwargs)

        live.update_progress(
            status="evaluating",
            algorithm=algorithm_name,
            run=run_idx,
            evaluation=eval_counter,
            max_evaluations=max_evaluations,
            completed_runs=completed_runs,
            total_runs=total_runs,
            current_config=config_data,
        )

        last_item_update = 0.0

        def on_item_progress(done: int, total: int) -> None:
            nonlocal last_item_update
            now = time.time()
            # Throttle progress writes to avoid excessive IO.
            if done != total and now - last_item_update < 2.0:
                return
            last_item_update = now
            live.update_progress(
                status="evaluating_items",
                algorithm=algorithm_name,
                run=run_idx,
                evaluation=eval_counter,
                evaluation_item=done,
                evaluation_total=total,
                max_evaluations=max_evaluations,
                completed_runs=completed_runs,
                total_runs=total_runs,
            )

        t0 = time.time()
        score = evaluate_rag_pipeline(
            config_dict=config_data,  # type: ignore[arg-type]
            progress=on_item_progress,
            trace={
                "algorithm": algorithm_name,
                "run": run_idx,
                "evaluation": eval_counter,
                "trace_file": str(
                    (live.qa_traces_dir / f"{algorithm_name.replace(' ', '_').lower()}_run{run_idx}_eval{eval_counter}.jsonl").absolute()
                ),
            },
        )
        elapsed = time.time() - t0

        live.log_evaluation(
            {
                "timestamp": _utc_now_iso(),
                "algorithm": algorithm_name,
                "run": run_idx,
                "evaluation": eval_counter,
                "max_evaluations": max_evaluations,
                "config": config_data,
                "score": score,
                "time_seconds": elapsed,
            }
        )
        live.update_progress(
            status="evaluated",
            algorithm=algorithm_name,
            run=run_idx,
            evaluation=eval_counter,
            max_evaluations=max_evaluations,
            last_score=score,
            last_eval_time_seconds=elapsed,
            completed_runs=completed_runs,
            total_runs=total_runs,
        )
        return score

    start_time = time.time()
    best_config, best_score = algorithm_fn(
        SEARCH_SPACE,
        max_evaluations,
        evaluator=logged_evaluator,
    )
    elapsed = time.time() - start_time
    
    record = {
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
    live.log_run(record)
    return record


def run_experiment_sequential(
    algorithm_name: str, 
    algorithm_fn: Callable,
    live: LiveLogger,
    completed_runs: int,
    total_runs: int,
    *,
    run_indices: Optional[List[int]] = None,
    num_runs: int = NUM_RUNS,
    max_evaluations: int = MAX_EVALUATIONS,
) -> List[Dict]:
    """Run multiple trials of an algorithm sequentially."""
    records = []

    if run_indices is None:
        run_indices = list(range(1, num_runs + 1))

    for local_idx, run_idx in enumerate(run_indices, start=1):
        live.update_progress(
            status="run_start",
            algorithm=algorithm_name,
            run=run_idx,
            completed_runs=completed_runs,
            total_runs=total_runs,
        )
        result = run_single_trial(
            algorithm_name,
            algorithm_fn,
            run_idx,
            max_evaluations,
            live=live,
            completed_runs=completed_runs,
            total_runs=total_runs,
        )
        records.append(result)
        completed_runs += 1

        config_str = (
            f"chunk={result['chunk_size']}, overlap={result['chunk_overlap']}, "
            f"top_k={result['top_k']}, thr={result['similarity_threshold']}, "
            f"metric={result['retrieval_metric']}, embed={result['embedding_model']}"
        )
        print(
            f"[{algorithm_name}] Run {run_idx}/{num_runs} "
            f"(segment {local_idx}/{len(run_indices)}) "
            f"=> score={result['best_score']:.4f} "
            f"({config_str}) "
            f"[{result['time_seconds']:.1f}s]"
        )
        live.update_progress(
            status="run_complete",
            algorithm=algorithm_name,
            run=run_idx,
            completed_runs=completed_runs,
            total_runs=total_runs,
            last_best_score=result["best_score"],
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


def main():
    """Main entry point."""
    baseline_only = _env_bool("RAG_BASELINE_ONLY", False)
    try:
        selected_algorithms = _select_algorithms()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    run_start = max(1, _env_int("RAG_RUN_START", 1))
    run_end = _env_int("RAG_RUN_END", NUM_RUNS)
    run_end = min(NUM_RUNS, max(1, run_end))
    if run_start > run_end:
        print(
            f"Invalid run range: RAG_RUN_START={run_start} > RAG_RUN_END={run_end}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    run_indices = list(range(run_start, run_end + 1))
    total_runs = len(selected_algorithms) * len(run_indices)
    live = LiveLogger(live_dir=LIVE_DIR, results_csv=RESULTS_DIR / "experiment_results.csv")
    live.reset(total_runs=total_runs, max_evaluations=MAX_EVALUATIONS)

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
    print(f"  - Number of runs: {NUM_RUNS} (executing {run_start}..{run_end})")
    print(f"  - Algorithms: {', '.join(selected_algorithms.keys())}")
    print(f"  - Results dir: {RESULTS_DIR}")
    print("=" * 80)
    
    # Initialize evaluator (pre-computes embeddings)
    print("\nInitializing high-performance evaluator...")
    start_time = time.time()
    evaluator = get_evaluator()
    evaluator.clear_all_caches()
    init_time = time.time() - start_time
    print(f"Initialization complete in {init_time:.1f}s")
    print(f"  • Dataset size: {len(evaluator.dataset)}")
    print(f"  • Eval sample: {len(evaluator.eval_dataset)}")
    print(f"  • Documents: {len(evaluator.documents)}")
    print(f"  • Parallel workers: {evaluator.num_workers}")
    print(f"  • Retriever: {evaluator.retriever_type} | Generator: {'llm' if evaluator.use_llm else 'extractive'}")
    live.update_progress(
        status="evaluator_ready",
        dataset_size=len(evaluator.dataset),
        eval_sample_size=len(evaluator.eval_dataset),
        documents=len(evaluator.documents),
        retriever=evaluator.retriever_type,
        generator=("llm" if evaluator.use_llm else "extractive"),
    )

    # Baseline: LLM without retrieval (only meaningful when generator is LLM).
    baselines: Dict[str, object] = {}
    if evaluator.use_llm:
        print("\n==================================================")
        print("Running No-RAG LLM baseline...")
        print("==================================================")

        baseline_last_update = 0.0

        def baseline_progress(done: int, total: int) -> None:
            nonlocal baseline_last_update
            now = time.time()
            if done != total and now - baseline_last_update < 2.0:
                return
            baseline_last_update = now
            live.update_progress(
                status="baseline_no_rag",
                baseline="no_rag_llm",
                baseline_item=done,
                baseline_total=total,
            )

        baseline_start = time.time()
        baseline_score, _ = evaluator.evaluate_baseline_no_rag(
            return_details=False,
            progress=baseline_progress,
            trace={"baseline": "no_rag_llm"},
        )
        baseline_time = time.time() - baseline_start
        baselines["no_rag_llm_mean_fitness"] = float(baseline_score)
        baselines["no_rag_llm_time_seconds"] = float(baseline_time)
        (RESULTS_DIR / "baselines.json").write_text(
            json.dumps(baselines, indent=2),
            encoding="utf-8",
        )
        print(f"No-RAG baseline mean fitness: {baseline_score:.4f} [{baseline_time:.1f}s]")
        live.update_progress(
            status="baseline_complete",
            baselines=baselines,
        )
        if baseline_only:
            print("\nBaseline-only run complete (RAG_BASELINE_ONLY=1).")
            print(f"Baseline saved to: {(RESULTS_DIR / 'baselines.json').absolute()}")
            return
    elif baseline_only:
        print(
            "\nRAG_BASELINE_ONLY=1 requested, but the current generator is not LLM "
            "(set RAG_GENERATOR=seq2seq to enable the No-RAG LLM baseline).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    
    # Run experiments
    print("\n" + "=" * 80)
    print("Running experiments...")
    print("=" * 80)
    
    experiment_start = time.time()
    
    # Sequential mode (more reliable, still fast due to caching)
    all_records = []
    completed_runs = 0
    selected_algorithm_names = list(selected_algorithms.keys())
    for algorithm_name, algorithm_fn in selected_algorithms.items():
        print(f"\n{'='*50}")
        print(f"Running {algorithm_name}...")
        print(f"{'='*50}")
        records = run_experiment_sequential(
            algorithm_name,
            algorithm_fn,
            live=live,
            completed_runs=completed_runs,
            total_runs=total_runs,
            run_indices=run_indices,
            num_runs=NUM_RUNS,
        )
        completed_runs += len(records)
        all_records.extend(records)
    
    total_time = time.time() - experiment_start
    
    # Save and report
    df = save_results(all_records, results_dir=RESULTS_DIR)

    skip_plots = _env_bool("RAG_SKIP_PLOTS", False)
    skip_summary = _env_bool("RAG_SKIP_SUMMARY", False)
    plots_dir = RESULTS_DIR / "plots"
    if not skip_plots:
        plot_results(df, plots_dir=plots_dir, algorithms=selected_algorithm_names)
    if not skip_summary:
        export_summary(df, results_dir=RESULTS_DIR, algorithms=selected_algorithm_names, baselines=baselines)
    
    # Final timing
    minutes, secs = divmod(int(total_time), 60)
    hours, minutes = divmod(minutes, 60)
    time_str = f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"
    print(f"\nTotal experiment time: {time_str}")
    print(f"Results saved to: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
