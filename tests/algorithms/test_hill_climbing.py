from src.algorithms.hill_climbing import hill_climbing

# Define the search space for testing
SEARCH_SPACE = {
    "chunk_size": [128, 256, 512],
    "top_k": [1, 2, 3],
}

MAX_CHUNK = max(SEARCH_SPACE["chunk_size"])
MAX_TOP_K = max(SEARCH_SPACE["top_k"])


def dummy_evaluator(chunk_size, top_k):
    normalized_chunk = chunk_size / MAX_CHUNK
    normalized_top_k = top_k / MAX_TOP_K
    return (normalized_chunk + normalized_top_k) / 2


def test_hill_climbing():
    """
    Tests if the hill climbing algorithm returns a configuration and a score.
    """
    config, score = hill_climbing(
        SEARCH_SPACE,
        max_evaluations=5,
        evaluator=dummy_evaluator,
    )

    assert "chunk_size" in config
    assert "top_k" in config
    assert config["chunk_size"] in SEARCH_SPACE["chunk_size"]
    assert config["top_k"] in SEARCH_SPACE["top_k"]
    assert 0 <= score <= 1
