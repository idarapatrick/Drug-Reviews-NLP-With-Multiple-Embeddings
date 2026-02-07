"""
Embedding module initialization and utilities.

Provides access to all available embeddings and utilities for working with them.
"""

from .tfidf_embedding import TfidfEmbedding, get_tfidf_embedding
from .word2vec_embedding import Word2VecEmbedding, get_word2vec_embedding
from .glove_embedding import GloVeEmbedding, get_glove_embedding

__all__ = [
    "TfidfEmbedding",
    "Word2VecEmbedding",
    "GloVeEmbedding",
    "get_tfidf_embedding",
    "get_word2vec_embedding",
    "get_glove_embedding",
    "get_embedding",
]

# Embedding registry
EMBEDDINGS = {
    "tfidf": {
        "class": TfidfEmbedding,
        "get_fn": get_tfidf_embedding,
        "configs": ["conservative", "balanced", "aggressive"],
        "requires_tokenization": False,
    },
    "word2vec": {
        "class": Word2VecEmbedding,
        "get_fn": get_word2vec_embedding,
        "configs": [
            "skipgram_small", "skipgram_medium", "skipgram_large",
            "cbow_small", "cbow_medium", "cbow_large",
        ],
        "requires_tokenization": True,
    },
    "glove": {
        "class": GloVeEmbedding,
        "get_fn": get_glove_embedding,
        "configs": ["small", "medium", "large"],
        "requires_tokenization": True,
    },
}


def get_embedding(embedding_name: str, config_name: str = None):
    """
    Get an embedding instance by name and configuration.
    
    Args:
        embedding_name: One of 'tfidf', 'word2vec', 'glove'
        config_name: Configuration name for the embedding (optional)
        
    Returns:
        Embedding instance
        
    Example:
        >>> embed = get_embedding('word2vec', 'skipgram_medium')
        >>> embed.fit(texts)
    """
    if embedding_name not in EMBEDDINGS:
        raise ValueError(
            f"Unknown embedding: {embedding_name}. "
            f"Available: {list(EMBEDDINGS.keys())}"
        )
    
    embedding_info = EMBEDDINGS[embedding_name]
    get_fn = embedding_info["get_fn"]
    
    if config_name is None:
        # Use default config
        if embedding_name == "tfidf":
            config_name = "balanced"
        elif embedding_name == "word2vec":
            config_name = "skipgram_medium"
        elif embedding_name == "glove":
            config_name = "medium"
    
    if config_name not in embedding_info["configs"]:
        raise ValueError(
            f"Unknown config for {embedding_name}: {config_name}. "
            f"Available: {embedding_info['configs']}"
        )
    
    return get_fn(config_name)


def get_embedding_info() -> dict:
    """
    Get information about all available embeddings.
    
    Returns:
        Dictionary with embedding information
    """
    info = {}
    for embedding_name, embedding_info in EMBEDDINGS.items():
        info[embedding_name] = {
            "class": embedding_info["class"].__name__,
            "configs": embedding_info["configs"],
            "requires_tokenization": embedding_info["requires_tokenization"],
        }
    return info
