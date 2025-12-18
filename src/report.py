from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.reporting import RUN_COLUMNS, export_summary, plot_results, save_results


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _load_baselines(dir_path: Path) -> Optional[Dict[str, Any]]:
    p = dir_path / "baselines.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_run_records(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Prefer live/runs.jsonl (append-only). Fall back to experiment_results.csv.
    """
    runs_path = input_dir / "live" / "runs.jsonl"
    runs = _read_jsonl(runs_path)
    if runs:
        # Ensure schema compatibility
        normalized: List[Dict[str, Any]] = []
        for r in runs:
            normalized.append({k: r.get(k) for k in RUN_COLUMNS})
        return normalized

    csv_path = input_dir / "experiment_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return []


def _dedupe(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """
    Dedupe by (algorithm, run) keeping the highest best_score.
    Returns (deduped_records, duplicates_found).
    """
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    duplicates: List[Tuple[str, int]] = []
    for r in records:
        alg = str(r.get("algorithm"))
        run = int(r.get("run"))
        key = (alg, run)
        score = float(r.get("best_score", float("-inf")))
        if key in best:
            duplicates.append(key)
            if score > float(best[key].get("best_score", float("-inf"))):
                best[key] = r
        else:
            best[key] = r
    deduped = list(best.values())
    return deduped, duplicates


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.report",
        description="Merge distributed run results and generate plots/summary.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Result directories (each containing live/runs.jsonl or experiment_results.csv).",
    )
    parser.add_argument(
        "--out",
        default="results/combined",
        help="Output directory for merged report (default: results/combined).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Dedupe by (algorithm, run), keeping the best score.",
    )
    args = parser.parse_args(argv)

    if not args.inputs:
        print("No input directories provided.", file=sys.stderr)
        print("Example: python -m src.report results/random_search results/hill_climbing results/simulated_annealing", file=sys.stderr)
        return 2

    input_dirs = [Path(p).expanduser() for p in args.inputs]
    out_dir = Path(args.out).expanduser()

    all_records: List[Dict[str, Any]] = []
    baselines_by_dir: Dict[str, Any] = {}

    for d in input_dirs:
        if not d.exists():
            print(f"Skipping missing dir: {d}", file=sys.stderr)
            continue
        all_records.extend(_collect_run_records(d))
        b = _load_baselines(d)
        if b is not None:
            baselines_by_dir[str(d)] = b

    if not all_records:
        print("No run records found in the provided directories.", file=sys.stderr)
        return 1

    if args.dedupe:
        all_records, duplicates = _dedupe(all_records)
        if duplicates:
            uniq = sorted(set(duplicates))
            print(f"Deduped duplicate run keys: {uniq}", file=sys.stderr)

    df = save_results(all_records, results_dir=out_dir)
    algorithms = sorted({str(a) for a in df["algorithm"].tolist() if pd.notna(a)})

    baselines: Dict[str, Any] = {}
    if baselines_by_dir:
        baselines["_baselines_by_dir"] = baselines_by_dir
        # Optional convenience: if all baselines have the same key, compute the mean.
        values = []
        for entry in baselines_by_dir.values():
            v = entry.get("no_rag_llm_mean_fitness")
            if isinstance(v, (int, float)):
                values.append(float(v))
        if values:
            baselines["no_rag_llm_mean_fitness_mean"] = sum(values) / len(values)

    plot_results(df, plots_dir=out_dir / "plots", algorithms=algorithms)
    export_summary(df, results_dir=out_dir, algorithms=algorithms, baselines=(baselines if baselines else None))
    print(f"\nMerged report written to: {out_dir.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

