from src.rag.evaluator import evaluate_rag_pipeline


def test_evaluate_rag_pipeline():
    """
    Tests if the evaluator returns a score between 0 and 1.
    """
    score = evaluate_rag_pipeline(chunk_size=128, top_k=1)
    assert 0 <= score <= 1
