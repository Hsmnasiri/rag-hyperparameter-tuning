import os

os.environ.setdefault("RAG_RETRIEVER", "tfidf")
os.environ.setdefault("RAG_GENERATOR", "extractive")
os.environ.setdefault("RAG_DATASET_SIZE", "15")
os.environ.setdefault("RAG_EVAL_SAMPLE_SIZE", "5")
os.environ.setdefault("RAG_DATASET_PATH", "data/squad_dev_v2.0_subset.json")
