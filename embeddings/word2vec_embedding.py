"""
Word2Vec Embedding implementation (Skip-gram and CBOW).

Uses gensim for training Word2Vec models. Provides both word-level and
document-level embeddings suitable for neural network inputs.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import warnings


class Word2VecEmbedding:
    """Word2Vec embedding wrapper supporting both Skip-gram and CBOW."""
    
    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
        workers: int = 4,
        epochs: int = 5,
        negative_samples: int = 5,
        alpha: float = 0.025,
        min_alpha: float = 0.0001
    ):
        """
        Initialize Word2Vec embedding.
        
        Args:
            embedding_dim: Dimension of word vectors
            window_size: Context window size
            min_count: Minimum word frequency
            sg: 1 for Skip-gram, 0 for CBOW
            workers: Number of worker threads
            epochs: Number of training epochs
            negative_samples: Number of negative samples for negative sampling
            alpha: Initial learning rate
            min_alpha: Minimum learning rate
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg  # 1 = Skip-gram, 0 = CBOW
        self.workers = workers
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.alpha = alpha
        self.min_alpha = min_alpha
        
        self.model = None
        self.is_fitted = False
        self.model_type = "Skip-gram" if sg == 1 else "CBOW"
        
    def fit(self, texts: List[List[str]]) -> 'Word2VecEmbedding':
        """
        Train Word2Vec model on tokenized texts.
        
        Args:
            texts: List of tokenized sentences (list of list of words)
            
        Returns:
            Self for chaining
        """
        self.model = Word2Vec(
            sentences=texts,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs,
            negative=self.negative_samples,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            seed=42
        )
        self.is_fitted = True
        return self
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector for a single word.
        
        Args:
            word: Word string
            
        Returns:
            Word vector of shape (embedding_dim,)
            
        Raises:
            KeyError: If word not in vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word not in self.model.wv:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        return self.model.wv[word]
    
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
        
        if word in self.model.wv:
            return self.model.wv[word]
        
        if default is not None:
            return default
        
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def encode_text(self, tokens: List[str]) -> np.ndarray:
        """
        Encode a single text (list of tokens) to a vector by averaging word vectors.
        
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
        """
        Get vocabulary size.
        
        Returns:
            Number of words in vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return len(self.model.wv)
    
    def get_vocab(self) -> List[str]:
        """
        Get list of all words in vocabulary.
        
        Returns:
            List of vocabulary words
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return list(self.model.wv.index_to_key)
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Get most similar words to a given word.
        
        Args:
            word: Query word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word not in self.model.wv:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        return self.model.wv.most_similar(word, topn=topn)
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if word1 not in self.model.wv or word2 not in self.model.wv:
            raise KeyError("One or both words not in vocabulary")
        
        return self.model.wv.similarity(word1, word2)
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> 'Word2VecEmbedding':
        """
        Load a pre-trained model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self for chaining
        """
        self.model = Word2Vec.load(filepath)
        self.is_fitted = True
        self.embedding_dim = self.model.vector_size
        return self
    
    def get_model_info(self) -> dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {
            "type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.get_vocabulary_size(),
            "window_size": self.window_size,
            "epochs": self.epochs,
        }


class Word2VecEmbeddingConfig:
    """Configuration templates for Word2Vec embeddings."""
    
    @staticmethod
    def skipgram_small() -> Tuple[dict, int]:
        """Small Skip-gram model for experiments."""
        return ({
            "embedding_dim": 100,
            "window_size": 5,
            "min_count": 2,
            "sg": 1,  # Skip-gram
            "epochs": 5,
            "negative_samples": 5,
        }, 1)
    
    @staticmethod
    def skipgram_medium() -> Tuple[dict, int]:
        """Medium Skip-gram model - balanced."""
        return ({
            "embedding_dim": 200,
            "window_size": 8,
            "min_count": 2,
            "sg": 1,  # Skip-gram
            "epochs": 10,
            "negative_samples": 10,
        }, 1)
    
    @staticmethod
    def skipgram_large() -> Tuple[dict, int]:
        """Large Skip-gram model for production."""
        return ({
            "embedding_dim": 300,
            "window_size": 10,
            "min_count": 1,
            "sg": 1,  # Skip-gram
            "epochs": 15,
            "negative_samples": 15,
        }, 1)
    
    @staticmethod
    def cbow_small() -> Tuple[dict, int]:
        """Small CBOW model for experiments."""
        return ({
            "embedding_dim": 100,
            "window_size": 5,
            "min_count": 2,
            "sg": 0,  # CBOW
            "epochs": 5,
            "negative_samples": 5,
        }, 0)
    
    @staticmethod
    def cbow_medium() -> Tuple[dict, int]:
        """Medium CBOW model - balanced."""
        return ({
            "embedding_dim": 200,
            "window_size": 8,
            "min_count": 2,
            "sg": 0,  # CBOW
            "epochs": 10,
            "negative_samples": 10,
        }, 0)
    
    @staticmethod
    def cbow_large() -> Tuple[dict, int]:
        """Large CBOW model for production."""
        return ({
            "embedding_dim": 300,
            "window_size": 10,
            "min_count": 1,
            "sg": 0,  # CBOW
            "epochs": 15,
            "negative_samples": 15,
        }, 0)


def get_word2vec_embedding(config_name: str = "skipgram_medium") -> Word2VecEmbedding:
    """
    Get a preconfigured Word2Vec embedding.
    
    Args:
        config_name: One of 'skipgram_small', 'skipgram_medium', 'skipgram_large',
                     'cbow_small', 'cbow_medium', 'cbow_large'
        
    Returns:
        Configured Word2VecEmbedding instance
    """
    configs = {
        "skipgram_small": Word2VecEmbeddingConfig.skipgram_small,
        "skipgram_medium": Word2VecEmbeddingConfig.skipgram_medium,
        "skipgram_large": Word2VecEmbeddingConfig.skipgram_large,
        "cbow_small": Word2VecEmbeddingConfig.cbow_small,
        "cbow_medium": Word2VecEmbeddingConfig.cbow_medium,
        "cbow_large": Word2VecEmbeddingConfig.cbow_large,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
    
    config_dict, sg_type = configs[config_name]()
    return Word2VecEmbedding(**config_dict)
