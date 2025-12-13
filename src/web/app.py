"""
FastAPI Web Interface for RAG Hyperparameter Tuning Dashboard.

Run with: uvicorn src.web.app:app --reload --port 8000
Then open: http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.random_search import random_search
from src.algorithms.simulated_annealing import simulated_annealing
from src.rag.evaluator import evaluate_rag_pipeline, get_evaluator
from src.rag.search_space import DEFAULT_SEARCH_SPACE

app = FastAPI(title="RAG Hyperparameter Tuning Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

ALGORITHMS = {
    "random_search": random_search,
    "hill_climbing": hill_climbing,
    "simulated_annealing": simulated_annealing,
}

SEARCH_SPACE = DEFAULT_SEARCH_SPACE.get_config_space()

executor = ThreadPoolExecutor(max_workers=2)


class ExperimentConfig(BaseModel):
    algorithm: str = "random_search"
    max_evaluations: int = 25
    num_runs: int = 5


class SingleEvalRequest(BaseModel):
    chunk_size: int
    chunk_overlap: int = SEARCH_SPACE["chunk_overlap"][0]
    top_k: int = SEARCH_SPACE["top_k"][0]
    similarity_threshold: float = SEARCH_SPACE["similarity_threshold"][0]
    retrieval_metric: str = SEARCH_SPACE["retrieval_metric"][0]
    embedding_model: str = SEARCH_SPACE["embedding_model"][0]
    max_context_chars: int = SEARCH_SPACE["max_context_chars"][0]


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, get_evaluator)


@app.get("/", response_class=HTMLResponse)
async def root():
    return get_dashboard_html()


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/results")
async def get_results():
    results_file = RESULTS_DIR / "experiment_results.csv"
    if not results_file.exists():
        return {"raw_data": [], "summary": {}}
    
    import pandas as pd
    df = pd.read_csv(results_file)
    
    summary = {}
    for algo in df["algorithm"].unique():
        scores = df[df["algorithm"] == algo]["best_score"].values
        summary[algo] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "max": float(scores.max()),
            "min": float(scores.min()),
        }
    
    return {"raw_data": df.to_dict(orient="records"), "summary": summary}


@app.post("/api/evaluate")
async def evaluate_single(request: SingleEvalRequest):
    loop = asyncio.get_event_loop()
    score = await loop.run_in_executor(
        executor,
        lambda: evaluate_rag_pipeline(request.dict()),
    )
    return {"config": request.dict(), "score": score}


@app.websocket("/ws/experiment")
async def websocket_experiment(websocket: WebSocket):
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        config = ExperimentConfig(**data)
        
        algorithms_to_run = list(ALGORITHMS.keys()) if config.algorithm == "all" else [config.algorithm]
        all_results = []
        total_runs = len(algorithms_to_run) * config.num_runs
        completed = 0
        
        for algo_name in algorithms_to_run:
            algo_fn = ALGORITHMS[algo_name]
            
            for run_idx in range(1, config.num_runs + 1):
                random.seed(run_idx * 42)
                
                await websocket.send_json({
                    "type": "progress",
                    "algorithm": algo_name,
                    "run": run_idx,
                    "total_runs": config.num_runs,
                    "overall_progress": completed / total_runs * 100,
                })
                
                loop = asyncio.get_event_loop()
                best_config, best_score = await loop.run_in_executor(
                    executor,
                    lambda a=algo_fn: a(SEARCH_SPACE, config.max_evaluations, evaluator=evaluate_rag_pipeline),
                )
                
                result = {
                    "algorithm": algo_name,
                    "run": run_idx,
                    "best_score": best_score,
                    "chunk_size": best_config["chunk_size"],
                    "top_k": best_config["top_k"],
                    "chunk_overlap": best_config.get("chunk_overlap"),
                    "similarity_threshold": best_config.get("similarity_threshold"),
                    "retrieval_metric": best_config.get("retrieval_metric"),
                    "embedding_model": best_config.get("embedding_model"),
                    "max_context_chars": best_config.get("max_context_chars"),
                }
                all_results.append(result)
                completed += 1
                
                await websocket.send_json({
                    "type": "result",
                    "data": result,
                    "overall_progress": completed / total_runs * 100,
                })
        
        await websocket.send_json({"type": "complete", "results": all_results})
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


def get_dashboard_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Hyperparameter Tuning</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(12px); }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 min-h-screen text-white">
    <div class="max-w-7xl mx-auto px-6 py-8">
        <div class="flex items-center justify-between mb-8">
            <div>
                <h1 class="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                    RAG Hyperparameter Tuning
                </h1>
                <p class="text-slate-400 mt-1">Search-Based Software Engineering</p>
            </div>
            <div id="status" class="flex items-center gap-2 px-4 py-2 glass rounded-full border border-slate-700">
                <div class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span class="text-sm text-emerald-400">Ready</span>
            </div>
        </div>

        <div class="grid grid-cols-3 gap-6">
            <div class="space-y-6">
                <div class="glass rounded-2xl border border-slate-700 p-6">
                    <h2 class="text-lg font-semibold mb-4">Settings</h2>
                    <div class="space-y-4">
                        <div>
                            <label class="text-sm text-slate-400 block mb-2">Algorithm</label>
                            <select id="algorithm" class="w-full bg-slate-700 border border-slate-600 rounded-xl px-4 py-3">
                                <option value="all">All Algorithms</option>
                                <option value="random_search">Random Search</option>
                                <option value="hill_climbing">Hill Climbing</option>
                                <option value="simulated_annealing">Simulated Annealing</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-sm text-slate-400 block mb-2">Evaluations: <span id="evalValue">25</span></label>
                            <input type="range" id="maxEvaluations" min="10" max="50" value="25" class="w-full accent-indigo-500">
                        </div>
                        <div>
                            <label class="text-sm text-slate-400 block mb-2">Runs: <span id="runsValue">5</span></label>
                            <input type="range" id="numRuns" min="1" max="10" value="5" class="w-full accent-indigo-500">
                        </div>
                    </div>
                </div>
                
                <button id="runBtn" onclick="runExperiment()" class="w-full py-4 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl font-semibold flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/></svg>
                    Run Experiment
                </button>
                
                <div class="glass rounded-2xl border border-slate-700 p-6">
                    <h2 class="text-lg font-semibold mb-4">Quick Test</h2>
                    <div class="grid grid-cols-2 gap-3 mb-4">
                        <div>
                            <label class="text-sm text-slate-400 block mb-1">Chunk Size</label>
                            <input type="number" id="singleChunk" value="256" min="128" max="1024" step="64" class="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2">
                        </div>
                        <div>
                            <label class="text-sm text-slate-400 block mb-1">Top K</label>
                            <input type="number" id="singleTopK" value="5" min="1" max="10" class="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2">
                        </div>
                    </div>
                    <button onclick="evaluateSingle()" class="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition">Evaluate</button>
                    <div id="singleResult" class="mt-3 text-center text-sm text-slate-400"></div>
                </div>
            </div>

            <div class="col-span-2 space-y-6">
                <div id="progressSection" class="hidden glass rounded-2xl border border-slate-700 p-6">
                    <div class="flex justify-between mb-3">
                        <h3 class="font-semibold">Progress</h3>
                        <span id="progressPercent" class="text-indigo-400 font-mono">0%</span>
                    </div>
                    <div class="h-3 bg-slate-700 rounded-full overflow-hidden">
                        <div id="progressBar" class="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all" style="width:0%"></div>
                    </div>
                    <p id="progressText" class="mt-2 text-sm text-slate-400">Starting...</p>
                </div>

                <div class="grid grid-cols-2 gap-6">
                    <div class="glass rounded-2xl border border-slate-700 p-6">
                        <h3 class="font-semibold mb-4">Algorithm Comparison</h3>
                        <canvas id="comparisonChart"></canvas>
                    </div>
                    <div class="glass rounded-2xl border border-slate-700 p-6">
                        <h3 class="font-semibold mb-4">Score by Run</h3>
                        <canvas id="runChart"></canvas>
                    </div>
                </div>

                <div class="glass rounded-2xl border border-slate-700 p-6">
                    <h3 class="font-semibold mb-4">Results</h3>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-slate-400 border-b border-slate-700">
                                <th class="text-left py-3">Algorithm</th>
                                <th class="text-left py-3">Mean</th>
                                <th class="text-left py-3">Std</th>
                                <th class="text-left py-3">Best</th>
                            </tr>
                        </thead>
                        <tbody id="resultsBody">
                            <tr><td colspan="4" class="py-8 text-center text-slate-500">No results yet</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let compChart, runChart, results = [];
        const colors = {random_search:'#6366f1', hill_climbing:'#10b981', simulated_annealing:'#f59e0b'};
        const names = {random_search:'Random Search', hill_climbing:'Hill Climbing', simulated_annealing:'Simulated Annealing'};

        document.getElementById('maxEvaluations').oninput = e => document.getElementById('evalValue').textContent = e.target.value;
        document.getElementById('numRuns').oninput = e => document.getElementById('runsValue').textContent = e.target.value;

        function initCharts() {
            compChart = new Chart(document.getElementById('comparisonChart'), {
                type: 'bar',
                data: {labels: [], datasets: [{label: 'Mean Score', data: [], backgroundColor: []}]},
                options: {responsive: true, scales: {y: {beginAtZero: false, min: 0, max: 1}}}
            });
            runChart = new Chart(document.getElementById('runChart'), {
                type: 'line',
                data: {labels: [], datasets: []},
                options: {responsive: true, scales: {y: {beginAtZero: false, min: 0, max: 1}}}
            });
        }

        function updateCharts() {
            const algos = [...new Set(results.map(r => r.algorithm))];
            const means = algos.map(a => {
                const scores = results.filter(r => r.algorithm === a).map(r => r.best_score);
                return scores.reduce((x,y) => x+y, 0) / scores.length;
            });
            compChart.data.labels = algos.map(a => names[a] || a);
            compChart.data.datasets[0].data = means;
            compChart.data.datasets[0].backgroundColor = algos.map(a => colors[a]);
            compChart.update();

            const runs = [...new Set(results.map(r => r.run))].sort((a,b) => a-b);
            runChart.data.labels = runs;
            runChart.data.datasets = algos.map(a => ({
                label: names[a] || a,
                data: runs.map(r => results.find(x => x.algorithm === a && x.run === r)?.best_score),
                borderColor: colors[a],
                tension: 0.3, fill: false
            }));
            runChart.update();
            updateTable();
        }

        function updateTable() {
            const algos = [...new Set(results.map(r => r.algorithm))];
            document.getElementById('resultsBody').innerHTML = algos.map(a => {
                const scores = results.filter(r => r.algorithm === a).map(r => r.best_score);
                const mean = scores.reduce((x,y) => x+y, 0) / scores.length;
                const std = Math.sqrt(scores.map(s => (s-mean)**2).reduce((x,y) => x+y, 0) / scores.length);
                return `<tr class="border-b border-slate-700/50"><td class="py-3">${names[a]}</td><td class="font-mono text-indigo-400">${mean.toFixed(4)}</td><td class="font-mono text-slate-400">${std.toFixed(4)}</td><td class="font-mono text-emerald-400">${Math.max(...scores).toFixed(4)}</td></tr>`;
            }).join('') || '<tr><td colspan="4" class="py-8 text-center text-slate-500">No results</td></tr>';
        }

        async function runExperiment() {
            const config = {
                algorithm: document.getElementById('algorithm').value,
                max_evaluations: +document.getElementById('maxEvaluations').value,
                num_runs: +document.getElementById('numRuns').value
            };
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('runBtn').disabled = true;
            document.getElementById('runBtn').classList.add('opacity-50');
            results = [];

            const ws = new WebSocket(`ws://${location.host}/ws/experiment`);
            ws.onopen = () => {
                ws.send(JSON.stringify(config));
                document.getElementById('status').innerHTML = '<div class="w-2 h-2 bg-amber-400 rounded-full animate-pulse"></div><span class="text-sm text-amber-400">Running</span>';
            };
            ws.onmessage = e => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'progress') {
                    document.getElementById('progressBar').style.width = msg.overall_progress + '%';
                    document.getElementById('progressPercent').textContent = Math.round(msg.overall_progress) + '%';
                    document.getElementById('progressText').textContent = `${names[msg.algorithm] || msg.algorithm} - Run ${msg.run}/${msg.total_runs}`;
                } else if (msg.type === 'result') {
                    results.push(msg.data);
                    updateCharts();
                    document.getElementById('progressBar').style.width = msg.overall_progress + '%';
                    document.getElementById('progressPercent').textContent = Math.round(msg.overall_progress) + '%';
                } else if (msg.type === 'complete') {
                    document.getElementById('progressSection').classList.add('hidden');
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('runBtn').classList.remove('opacity-50');
                    document.getElementById('status').innerHTML = '<div class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div><span class="text-sm text-emerald-400">Complete</span>';
                } else if (msg.type === 'error') {
                    alert('Error: ' + msg.message);
                    document.getElementById('runBtn').disabled = false;
                    document.getElementById('runBtn').classList.remove('opacity-50');
                }
            };
            ws.onerror = () => {
                alert('Connection error');
                document.getElementById('runBtn').disabled = false;
                document.getElementById('runBtn').classList.remove('opacity-50');
            };
        }

        async function evaluateSingle() {
            document.getElementById('singleResult').innerHTML = '<span class="text-amber-400">Evaluating...</span>';
            try {
                const r = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({chunk_size: +document.getElementById('singleChunk').value, top_k: +document.getElementById('singleTopK').value})
                });
                const d = await r.json();
                document.getElementById('singleResult').innerHTML = `Score: <span class="text-emerald-400 font-mono">${d.score.toFixed(4)}</span>`;
            } catch {
                document.getElementById('singleResult').innerHTML = '<span class="text-red-400">Error</span>';
            }
        }

        async function loadResults() {
            try {
                const r = await fetch('/api/results');
                const d = await r.json();
                if (d.raw_data?.length) { results = d.raw_data; updateCharts(); }
            } catch {}
        }

        initCharts();
        loadResults();
    </script>
</body>
</html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
