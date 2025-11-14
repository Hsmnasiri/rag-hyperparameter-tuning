from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_PATTERN = re.compile(r"\b\w+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _ensure_text(document: object) -> str:
    if document is None:
        return ""
    if hasattr(document, "page_content"):
        return str(getattr(document, "page_content") or "")
    return str(document)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            yield chunk
        start += step


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extract_answer(question: str, contexts: Sequence[str]) -> str:
    question_tokens = _tokenize(question)
    question_token_set = set(question_tokens)
    if not question_token_set:
        return contexts[0][:200].strip() if contexts else ""

    best_sentence = ""
    best_score = 0.0

    for context in contexts:
        for sentence in _split_sentences(context):
            answer_tokens = _tokenize(sentence)
            if not answer_tokens:
                continue

            overlap = len(question_token_set & set(answer_tokens))
            if overlap == 0:
                continue

            precision = overlap / len(answer_tokens)
            recall = overlap / len(question_token_set)
            f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_score:
                best_score = f1
                best_sentence = sentence

    if best_sentence:
        return best_sentence.strip()

    return contexts[0][:200].strip() if contexts else ""


class BaseRetriever:
    def retrieve(self, question: str, top_k: int) -> List[str]:
        raise NotImplementedError


class TFIDFRetriever(BaseRetriever):
    def __init__(self, chunk_texts: Sequence[str]):
        self.chunk_texts = list(chunk_texts)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, question: str, top_k: int) -> List[str]:
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.chunk_vectors)[0]
        if not np.any(similarities):
            return self.chunk_texts[:1]

        k = min(top_k, len(self.chunk_texts))
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.chunk_texts[idx] for idx in top_indices]


class DenseRetriever(BaseRetriever):
    def __init__(self, chunk_texts: Sequence[str], embedding_model: str):
        self.chunk_texts = list(chunk_texts)
        self.encoder = SentenceTransformer(embedding_model)
        self.chunk_vectors = self.encoder.encode(
            self.chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, question: str, top_k: int) -> List[str]:
        question_vector = self.encoder.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.dot(self.chunk_vectors, question_vector)
        if not np.any(scores):
            return self.chunk_texts[:1]

        k = min(top_k, len(self.chunk_texts))
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunk_texts[idx] for idx in top_indices]


class BaseGenerator:
    def generate(self, question: str, contexts: Sequence[str]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    def generate(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return ""
        return _extract_answer(question, contexts)


class Seq2SeqGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        max_input_length: int = 512,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, temperature)
        self.max_input_length = max_input_length
        self._torch = torch

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        context_block = "\n\n".join(contexts).strip()
        if not context_block:
            return ""

        prompt = (
            "You are an assistant that answers questions using the provided context.\n"
            "If the context does not contain the answer, respond with 'I don't know.'\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 4,
        }
        if self.temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "num_beams": 1,
                }
            )

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()


@dataclass
class RAGPipeline:
    retriever: BaseRetriever
    generator: BaseGenerator
    max_context_chars: int = 4096

    def _prepare_contexts(self, raw_contexts: Sequence[str]) -> List[str]:
        truncated_contexts: List[str] = []
        total = 0
        for ctx in raw_contexts:
            ctx = ctx.strip()
            if not ctx:
                continue
            remaining = self.max_context_chars - total
            if remaining <= 0:
                break
            truncated_contexts.append(ctx[:remaining])
            total += len(ctx[:remaining])
        return truncated_contexts

    def invoke(self, question: str, top_k: int, return_contexts: bool = False):
        question = question.strip()
        if not question:
            return ("", []) if return_contexts else ""

        contexts = self.retriever.retrieve(question, top_k)
        if not contexts:
            return ("", []) if return_contexts else ""

        truncated_contexts = self._prepare_contexts(contexts)
        if not truncated_contexts:
            return ("", []) if return_contexts else ""

        answer = self.generator.generate(question, truncated_contexts)
        if return_contexts:
            return answer, truncated_contexts
        return answer


def create_rag_pipeline(
    documents: Sequence[object],
    chunk_size: int,
    chunk_overlap: int = 64,
    retriever_type: str = "dense",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator_type: str = "seq2seq",
    generator_model: str = "google/flan-t5-small",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    max_context_chars: int = 4096,
) -> RAGPipeline:
    chunk_texts: List[str] = []
    for document in documents:
        text = _ensure_text(document)
        if not text.strip():
            continue
        chunk_texts.extend(_chunk_text(text, chunk_size, chunk_overlap))

    if not chunk_texts:
        raise ValueError("No non-empty document chunks were generated.")

    retriever_type = retriever_type.lower()
    if retriever_type == "tfidf":
        retriever = TFIDFRetriever(chunk_texts)
    elif retriever_type == "dense":
        retriever = DenseRetriever(chunk_texts, embedding_model=embedding_model)
    else:
        raise ValueError("retriever_type must be 'tfidf' or 'dense'.")

    generator_type = generator_type.lower()
    if generator_type == "extractive":
        generator = ExtractiveGenerator()
    elif generator_type == "seq2seq":
        generator = Seq2SeqGenerator(
            model_name=generator_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError("generator_type must be 'extractive' or 'seq2seq'.")

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        max_context_chars=max_context_chars,
    )
