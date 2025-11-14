from src.rag.pipeline import create_rag_pipeline

DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Ottawa is the capital city of Canada.",
]


def test_create_rag_pipeline():
    rag_chain = create_rag_pipeline(
        DOCUMENTS,
        chunk_size=64,
        retriever_type="tfidf",
        generator_type="extractive",
    )
    result = rag_chain.invoke("What is the capital of France?", top_k=1)
    assert isinstance(result, str)
    assert "France" in result or "Paris" in result
