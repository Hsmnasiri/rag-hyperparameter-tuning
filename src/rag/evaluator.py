from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .pipeline import RAGPipeline, create_rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULL_SQUAD_PATH = DATA_DIR / "squad_dev_v2.0.json"
SUBSET_SQUAD_PATH = DATA_DIR / "squad_dev_v2.0_subset.json"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

DATASET_PATH = Path(
    os.environ.get("RAG_DATASET_PATH", str(FULL_SQUAD_PATH))
).expanduser()
DEFAULT_DATASET_SIZE = int(os.environ.get("RAG_DATASET_SIZE", "500"))
DEFAULT_EVAL_SAMPLE_SIZE = max(
    1,
    min(
        int(os.environ.get("RAG_EVAL_SAMPLE_SIZE", "80")),
        DEFAULT_DATASET_SIZE,
    ),
)
DEFAULT_RETRIEVER = os.environ.get("RAG_RETRIEVER", "dense").lower()
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_GENERATOR = os.environ.get("RAG_GENERATOR", "seq2seq").lower()
DEFAULT_GENERATOR_MODEL = os.environ.get(
    "RAG_GENERATOR_MODEL", "google/flan-t5-small"
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("RAG_GENERATOR_MAX_NEW_TOKENS", "64"))
DEFAULT_TEMPERATURE = float(os.environ.get("RAG_GENERATOR_TEMPERATURE", "0.0"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "4096"))
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _download_squad(target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SQUAD_URL, target_path)


def _resolve_dataset_path() -> Path:
    if DATASET_PATH.exists():
        return DATASET_PATH

    try:
        _download_squad(DATASET_PATH)
        return DATASET_PATH
    except Exception:
        if SUBSET_SQUAD_PATH.exists():
            return SUBSET_SQUAD_PATH
        raise FileNotFoundError(
            "Could not retrieve the SQuAD dataset and the subset file is missing."
        )


ACTIVE_DATA_PATH = _resolve_dataset_path()


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_for_embedding(text: str) -> str:
    tokens = _tokenize(text)
    return " ".join(tokens)


def _iter_dataset_entries(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r") as f:
        squad_data = json.load(f)

    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    continue
                yield {
                    "question": qa["question"],
                    "answer": answers[0]["text"],
                    "context": context,
                }


def _build_dataset(path: Path, max_items: int) -> List[Dict[str, str]]:
    return list(islice(_iter_dataset_entries(path), max_items))


def _lexical_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0

    overlap = len(set(pred_tokens) & set(truth_tokens))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


DATASET_FULL: List[Dict[str, str]] = _build_dataset(ACTIVE_DATA_PATH, DEFAULT_DATASET_SIZE)
if not DATASET_FULL:
    raise RuntimeError(
        f"No questions were loaded from {ACTIVE_DATA_PATH}. "
        "Ensure the dataset file is present and not empty."
    )

EVAL_SAMPLE_SIZE = min(DEFAULT_EVAL_SAMPLE_SIZE, len(DATASET_FULL))
EVAL_DATASET = DATASET_FULL[:EVAL_SAMPLE_SIZE]
DOCUMENTS: List[str] = list(dict.fromkeys(entry["context"] for entry in DATASET_FULL))


@lru_cache(maxsize=128)
def _get_cached_pipeline(
    chunk_size: int,
    retriever_type: str,
    embedding_model: str,
    generator_type: str,
    generator_model: str,
    max_new_tokens: int,
    temperature: float,
    max_context_chars: int,
) -> RAGPipeline:
    return create_rag_pipeline(
        DOCUMENTS,
        chunk_size=chunk_size,
        retriever_type=retriever_type,
        embedding_model=embedding_model,
        generator_type=generator_type,
        generator_model=generator_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_context_chars=max_context_chars,
    )


@lru_cache(maxsize=4)
def _get_similarity_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embedding_similarity(encoder, prediction: str, ground_truth: str) -> float:
    clean_pred = _normalize_for_embedding(prediction)
    clean_truth = _normalize_for_embedding(ground_truth)
    if not clean_pred or not clean_truth:
        return 0.0

    embeddings = encoder.encode(
        [clean_pred, clean_truth],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    # Map cosine [-1,1] to [0,1] for consistency with other scores.
    return max(0.0, min(1.0, (similarity + 1) / 2))


def _evaluate_dataset(
    rag_chain: RAGPipeline,
    dataset: Sequence[Dict[str, str]],
    top_k: int,
    embedding_model: str,
) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    encoder = getattr(rag_chain.retriever, "encoder", None) or _get_similarity_encoder(
        embedding_model
    )
    for item in dataset:
        prediction, contexts = rag_chain.invoke(
            item["question"],
            top_k=top_k,
            return_contexts=True,
        )
        lexical_f1 = _lexical_f1(prediction, item["answer"])
        embedding_score = _embedding_similarity(encoder, prediction, item["answer"])
        details.append(
            {
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                "retrieved_contexts": contexts,
                "lexical_f1": lexical_f1,
                "embedding_similarity": embedding_score,
            }
        )
    return details


def evaluate_rag_pipeline(
    chunk_size: int,
    top_k: int,
    retriever_type: str | None = None,
    embedding_model: str | None = None,
    generator_type: str | None = None,
    generator_model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    max_context_chars: int | None = None,
    use_full_dataset: bool = False,
) -> float:
    retriever = (retriever_type or DEFAULT_RETRIEVER).lower()
    embedding = embedding_model or DEFAULT_EMBEDDING_MODEL
    generator = (generator_type or DEFAULT_GENERATOR).lower()
    generator_model = generator_model or DEFAULT_GENERATOR_MODEL
    max_tokens = max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    context_chars = max_context_chars if max_context_chars is not None else DEFAULT_MAX_CONTEXT_CHARS

    rag_chain = _get_cached_pipeline(
        chunk_size,
        retriever,
        embedding,
        generator,
        generator_model,
        max_tokens,
        temp,
        context_chars,
    )
    dataset = DATASET_FULL if use_full_dataset else EVAL_DATASET
    details = _evaluate_dataset(rag_chain, dataset, top_k, embedding)
    return sum(item["embedding_similarity"] for item in details) / len(details)


def evaluate_rag_with_details(
    chunk_size: int,
    top_k: int,
    retriever_type: str | None = None,
    embedding_model: str | None = None,
    generator_type: str | None = None,
    generator_model: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    max_context_chars: int | None = None,
    use_full_dataset: bool = True,
) -> Tuple[float, List[Dict[str, object]]]:
    retriever = (retriever_type or DEFAULT_RETRIEVER).lower()
    embedding = embedding_model or DEFAULT_EMBEDDING_MODEL
    generator = (generator_type or DEFAULT_GENERATOR).lower()
    generator_model = generator_model or DEFAULT_GENERATOR_MODEL
    max_tokens = max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    context_chars = max_context_chars if max_context_chars is not None else DEFAULT_MAX_CONTEXT_CHARS

    rag_chain = _get_cached_pipeline(
        chunk_size,
        retriever,
        embedding,
        generator,
        generator_model,
        max_tokens,
        temp,
        context_chars,
    )
    dataset = DATASET_FULL if use_full_dataset else EVAL_DATASET
    details = _evaluate_dataset(rag_chain, dataset, top_k, embedding)
    mean_score = sum(item["embedding_similarity"] for item in details) / len(details)
    return mean_score, details
