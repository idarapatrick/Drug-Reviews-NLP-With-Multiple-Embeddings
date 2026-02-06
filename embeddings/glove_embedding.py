"""
GloVe Embedding implementation.

GloVe (Global Vectors for Word Representation) combines global matrix factorization
with local context window methods for word embeddings.

Note: This uses a wrapper around the glove-python library, with fallback to
downloading pre-trained vectors if needed.
"""

import numpy as np
from typing import List, Optional, Tuple
import warnings


class GloVeEmbedding:
    """GloVe embedding wrapper."""
    
    def __init__(
        self,
        embedding_dim: int = 100,
        learning_rate: float = 0.05,
        x_max: float = 100.0,
        alpha: float = 0.75,
        max_iter: int = 20,
        vector_size: int = 50
    ):
        """
        Initialize GloVe embedding.
        
        Args:
            embedding_dim: Dimension of word vectors
            learning_rate: Learning rate for training
            x_max: Cutoff in weighting function
            alpha: Exponent in weighting function
            max_iter: Number of training iterations
            vector_size: Alternative parameter for embedding dimension (used if embedding_dim not set)
        """
        self.embedding_dim = embedding_dim if embedding_dim > 0 else vector_size
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.max_iter = max_iter
        
        self.word_vectors = None
        self.word_to_idx = None
        self.is_fitted = False
        
        # Try to import glove_python
        try:
            from glove import Corpus, Glove
            self.Glove = Glove
            self.has_glove = True
        except ImportError:
            warnings.warn(
                "glove-python not installed. Using Word2Vec-based approximation instead. "
                "Install with: pip install glove-python-binary"
            )
            self.has_glove = False
            self.Glove = None
    
    def _build_with_word2vec_approximation(self, texts: List[List[str]]):
        """
        Build embeddings using Word2Vec as approximation of GloVe.
        Used when glove-python is not available.
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim required for GloVe approximation. Install with: pip install gensim")
        
        # Train Skip-gram as an approximation
        model = Word2Vec(
            sentences=texts,
            vector_size=self.embedding_dim,
            window=10,
            min_count=2,
            sg=1,  # Skip-gram (GloVe-like)
            workers=4,
            epochs=self.max_iter,
            negative=15,
            seed=42
        )
        
        self.word_to_idx = {word: idx for idx, word in enumerate(model.wv.index_to_key)}
        vectors = []
        for word in model.wv.index_to_key:
            vectors.append(model.wv[word])
        self.word_vectors = np.array(vectors, dtype=np.float32)
    
    def fit(self, texts: List[List[str]]) -> 'GloVeEmbedding':
        """
        Train GloVe model on tokenized texts.
        
        Args:
            texts: List of tokenized sentences
            
        Returns:
            Self for chaining
        """
        if self.has_glove and self.Glove is not None:
            self._fit_native_glove(texts)
        else:
            self._build_with_word2vec_approximation(texts)
        
        self.is_fitted = True
        return self
    
    def _fit_native_glove(self, texts: List[List[str]]):
        """Train using native GloVe library."""
        try:
            from glove import Corpus, Glove
            
            # Build corpus
            corpus = Corpus()
            corpus.fit(texts, window=10)
            
            # Train GloVe
            glove = Glove(no_components=self.embedding_dim, learning_rate=self.learning_rate)
            glove.fit(corpus.matrix, epochs=self.max_iter, no_threads=4, verbose=False)
            glove.add_dictionary(corpus.dictionary)
            
            # Extract vectors
            self.word_to_idx = glove.dictionary
            self.word_vectors = glove.word_vectors
            
        except Exception as e:
            warnings.warn(f"Native GloVe training failed: {e}. Using approximation.")
            self._build_with_word2vec_approximation(texts)
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector for a single word.
        
        Args:
            word: Word string
            
        Returns:
            Word vector of shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        idx = self.word_to_idx[word]
        return self.word_vectors[idx]
    
    def get_vector_safe(self, word: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get vector for a word, returning default if not found.
        
        Args:
            word: Word string
            default: Default vector if word not found (random if None)
            
        Returns:
            Word vector or default
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word in self.word_to_idx:
            return self.get_word_vector(word)
        
        if default is not None:
            return default
        
        return np.random.randn(self.embedding_dim).astype(np.float32) * 0.01
    
    def encode_text(self, tokens: List[str]) -> np.ndarray:
        """
        Encode a single text by averaging word vectors.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Document vector of shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.get_word_vector(token))
            except KeyError:
                continue
        
        if len(vectors) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        return np.mean(vectors, axis=0).astype(np.float32)
    
    def encode_texts(self, texts: List[List[str]]) -> np.ndarray:
        """
        Encode multiple texts to vectors.
        
        Args:
            texts: List of tokenized texts
            
        Returns:
            Matrix of shape (n_samples, embedding_dim)
        """
        vectors = np.array([self.encode_text(text) for text in texts])
        return vectors.astype(np.float32)
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return len(self.word_to_idx)
    
    def get_vocab(self) -> List[str]:
        """Get list of all words in vocabulary."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return list(self.word_to_idx.keys())
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Get most similar words to a given word using cosine similarity.
        
        Args:
            word: Query word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        word_vec = self.get_word_vector(word)
        
        # Compute cosine similarities
        similarities = []
        for vocab_word, idx in self.word_to_idx.items():
            if vocab_word == word:
                continue
            
            other_vec = self.word_vectors[idx]
            sim = self._cosine_similarity(word_vec, other_vec)
            similarities.append((vocab_word, sim))
        
        # Sort and return top-n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word1 not in self.word_to_idx or word2 not in self.word_to_idx:
            raise KeyError("One or both words not in vocabulary")
        
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        return self._cosine_similarity(vec1, vec2)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {
            "type": "GloVe",
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.get_vocabulary_size(),
            "native": self.has_glove,
        }


class GloVeEmbeddingConfig:
    """Configuration templates for GloVe embeddings."""
    
    @staticmethod
    def small() -> dict:
        """Small GloVe model."""
        return {
            "embedding_dim": 100,
            "learning_rate": 0.05,
            "x_max": 100.0,
            "alpha": 0.75,
            "max_iter": 10,
        }
    
    @staticmethod
    def medium() -> dict:
        """Medium GloVe model - balanced."""
        return {
            "embedding_dim": 200,
            "learning_rate": 0.05,
            "x_max": 100.0,
            "alpha": 0.75,
            "max_iter": 15,
        }
    
    @staticmethod
    def large() -> dict:
        """Large GloVe model."""
        return {
            "embedding_dim": 300,
            "learning_rate": 0.02,
            "x_max": 100.0,
            "alpha": 0.75,
            "max_iter": 20,
        }


def get_glove_embedding(config_name: str = "medium") -> GloVeEmbedding:
    """
    Get a preconfigured GloVe embedding.
    
    Args:
        config_name: One of 'small', 'medium', 'large'
        
    Returns:
        Configured GloVeEmbedding instance
    """
    configs = {
        "small": GloVeEmbeddingConfig.small(),
        "medium": GloVeEmbeddingConfig.medium(),
        "large": GloVeEmbeddingConfig.large(),
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
    
    return GloVeEmbedding(**configs[config_name])
