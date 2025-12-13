"""
RAG Pipeline with LLM-based Generation (Primary) and Extractive QA (Baseline).

This module implements a complete RAG pipeline with:
1. Configurable chunking (size, overlap)
2. Dense retrieval with multiple embedding models
3. Similarity threshold filtering
4. LLM-based answer generation (primary)
5. Extractive QA (baseline/fallback)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RAGConfig, EMBEDDING_MODELS, GENERATOR_MODELS


TOKEN_PATTERN = re.compile(r"\b\w+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    
    return chunks


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return TOKEN_PATTERN.findall(text.lower())


# =============================================================================
# RETRIEVER CLASSES
# =============================================================================

class DenseRetriever:
    """
    Dense retriever using sentence transformers.
    
    Supports multiple embedding models and retrieval metrics.
    """
    
    def __init__(
        self,
        documents: Sequence[str],
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str = "minilm",
        retrieval_metric: str = "cosine",
    ):
        from sentence_transformers import SentenceTransformer
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_metric = retrieval_metric
        
        # Get full model name
        model_info = EMBEDDING_MODELS.get(embedding_model, EMBEDDING_MODELS["minilm"])
        model_name = model_info["name"]
        
        # Load encoder
        self.encoder = SentenceTransformer(model_name)
        
        # Create chunks
        self.chunks: List[str] = []
        for doc in documents:
            if doc.strip():
                self.chunks.extend(chunk_text(doc, chunk_size, chunk_overlap))
        
        # Encode chunks
        if self.chunks:
            # Always keep the base embeddings unnormalized; we normalize on demand
            self.chunk_embeddings = self.encoder.encode(
                self.chunks,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                batch_size=64,
            )
            norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True) + 1e-12
            self.normalized_chunk_embeddings = self.chunk_embeddings / norms
        else:
            self.chunk_embeddings = np.array([])
            self.normalized_chunk_embeddings = np.array([])
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        retrieval_metric: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score (0.0 = no threshold)
        
        Returns:
            Tuple of (chunks, scores)
        """
        if not self.chunks:
            return [], []
        
        # Encode query
        metric = retrieval_metric or self.retrieval_metric
        query_embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        # Compute similarities
        if metric == "cosine":
            denom = np.linalg.norm(query_embedding) + 1e-12
            query_norm = query_embedding / denom
            scores = np.dot(self.normalized_chunk_embeddings, query_norm)
        else:  # dot product
            scores = np.dot(self.chunk_embeddings, query_embedding)
        
        # Apply threshold
        if similarity_threshold > 0:
            valid_mask = scores >= similarity_threshold
            if not np.any(valid_mask):
                # If no chunk passes threshold, return top-1 anyway
                top_idx = np.argmax(scores)
                return [self.chunks[top_idx]], [float(scores[top_idx])]
            
            valid_indices = np.where(valid_mask)[0]
            valid_scores = scores[valid_mask]
            
            # Sort by score
            sorted_order = np.argsort(valid_scores)[::-1][:top_k]
            selected_indices = valid_indices[sorted_order]
            selected_scores = valid_scores[sorted_order]
            
            return (
                [self.chunks[i] for i in selected_indices],
                [float(s) for s in selected_scores],
            )
        
        # No threshold - just get top-k
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        
        return (
            [self.chunks[i] for i in top_indices],
            [float(scores[i]) for i in top_indices],
        )


# =============================================================================
# GENERATOR CLASSES
# =============================================================================

class LLMGenerator:
    """
    LLM-based answer generator using Seq2Seq models (e.g., Flan-T5).
    
    This is the PRIMARY generator for RAG evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        max_input_length: int = 512,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Use CPU (user doesn't have GPU)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, temperature)
        self.max_input_length = max_input_length
    
    def generate(self, question: str, contexts: List[str]) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            question: The question to answer
            contexts: Retrieved context chunks
        
        Returns:
            Generated answer string
        """
        import torch
        
        if not contexts:
            return ""
        
        # Build prompt
        context_text = "\n\n".join(contexts[:5])  # Limit contexts
        prompt = f"Answer the question based on the context.\n\nContext: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            if self.temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    num_beams=1,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


class ExtractiveQA:
    """
    Extractive QA baseline - extracts answer from context without LLM.
    
    This is the BASELINE generator for comparison.
    """
    
    def __init__(self, encoder=None):
        self.encoder = encoder
    
    def extract(self, question: str, contexts: List[str]) -> str:
        """
        Extract the most relevant sentence as answer.
        
        Uses semantic similarity if encoder is available,
        otherwise falls back to lexical matching.
        """
        if not contexts:
            return ""
        
        combined = " ".join(contexts)
        sentences = split_sentences(combined)
        
        if not sentences:
            return contexts[0][:200].strip()
        
        # Use semantic similarity if encoder available
        if self.encoder is not None:
            q_emb = self.encoder.encode(
                question,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            s_embs = self.encoder.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            scores = np.dot(s_embs, q_emb)
            best_idx = np.argmax(scores)
            return sentences[best_idx].strip()
        
        # Fallback: lexical F1 matching
        q_tokens = set(tokenize(question))
        if not q_tokens:
            return sentences[0].strip()
        
        best_sentence = ""
        best_score = 0.0
        
        for sentence in sentences:
            s_tokens = tokenize(sentence)
            if not s_tokens:
                continue
            
            overlap = len(q_tokens & set(s_tokens))
            if overlap == 0:
                continue
            
            precision = overlap / len(s_tokens)
            recall = overlap / len(q_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_score:
                best_score = f1
                best_sentence = sentence
        
        return best_sentence.strip() if best_sentence else sentences[0].strip()


# =============================================================================
# RAG PIPELINE
# =============================================================================

@dataclass
class RAGPipeline:
    """
    Complete RAG pipeline with configurable components.
    
    Supports both LLM-based generation (primary) and extractive QA (baseline).
    """
    retriever: DenseRetriever
    llm_generator: Optional[LLMGenerator]
    extractive_qa: ExtractiveQA
    retrieval_metric: str = "cosine"
    max_context_chars: int = 2048
    use_llm: bool = True  # True = LLM generation, False = Extractive baseline
    
    def _truncate_contexts(self, contexts: List[str]) -> List[str]:
        """Truncate contexts to fit max_context_chars."""
        truncated = []
        total = 0
        
        for ctx in contexts:
            remaining = self.max_context_chars - total
            if remaining <= 0:
                break
            truncated.append(ctx[:remaining])
            total += len(truncated[-1])
        
        return truncated
    
    def invoke(
        self,
        question: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        return_details: bool = False,
    ):
        """
        Run the RAG pipeline.
        
        Args:
            question: Question to answer
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval
            return_details: If True, return (answer, contexts, scores)
        
        Returns:
            Answer string, or tuple with details if return_details=True
        """
        question = question.strip()
        if not question:
            return ("", [], []) if return_details else ""
        
        # Retrieve
        contexts, scores = self.retriever.retrieve(
            question, top_k, similarity_threshold, retrieval_metric=self.retrieval_metric
        )
        
        if not contexts:
            return ("", [], []) if return_details else ""
        
        # Truncate
        contexts = self._truncate_contexts(contexts)
        
        # Generate answer
        if self.use_llm and self.llm_generator is not None:
            answer = self.llm_generator.generate(question, contexts)
        else:
            answer = self.extractive_qa.extract(question, contexts)
        
        if return_details:
            return answer, contexts, scores
        return answer


def create_rag_pipeline(
    documents: Sequence[str],
    config: Optional[RAGConfig] = None,
    use_llm: bool = True,
    preloaded_generator: Optional[LLMGenerator] = None,
    **legacy_kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline from configuration.
    
    Args:
        documents: List of document texts
        config: RAG configuration
        use_llm: Whether to use LLM generation (True) or extractive baseline (False)
        preloaded_generator: Optional pre-loaded LLM generator (for efficiency)
        legacy_kwargs: Backwards-compatible params (chunk_size, top_k, generator_type, etc.)
    
    Returns:
        Configured RAGPipeline instance
    """
    # Backwards compatibility: allow callers to pass raw kwargs instead of RAGConfig
    if config is None:
        config = RAGConfig.from_dict(legacy_kwargs)
    elif isinstance(config, dict):
        config = RAGConfig.from_dict(config)

    # Legacy toggle for extractive-only generation
    generator_type = legacy_kwargs.get("generator_type")
    if generator_type and generator_type.lower() == "extractive":
        use_llm = False
    # Create retriever
    retriever = DenseRetriever(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        retrieval_metric=config.retrieval_metric,
    )
    
    # Create or reuse LLM generator
    if use_llm:
        if preloaded_generator is not None:
            llm_generator = preloaded_generator
        else:
            llm_generator = LLMGenerator(
                model_name=config.get_generator_model_name(),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
    else:
        llm_generator = None
    
    # Create extractive QA (uses retriever's encoder)
    extractive_qa = ExtractiveQA(encoder=retriever.encoder)
    
    return RAGPipeline(
        retriever=retriever,
        llm_generator=llm_generator,
        extractive_qa=extractive_qa,
        retrieval_metric=config.retrieval_metric,
        max_context_chars=config.max_context_chars,
        use_llm=use_llm,
    )


# =============================================================================
# BASELINE: NO-RAG LLM
# =============================================================================

class NoRAGBaseline:
    """
    Baseline LLM without retrieval - for comparison.
    
    This shows the value added by RAG over pure LLM.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.generator = LLMGenerator(model_name=model_name)
    
    def answer(self, question: str) -> str:
        """Answer question without any context (pure LLM)."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        prompt = f"Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.generator.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.generator.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.generator.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )
        
        return self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
