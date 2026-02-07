"""
TF-IDF Embedding implementation.

Provides vectorization of text using TF-IDF (Term Frequency-Inverse Document Frequency).
Suitable for traditional ML models and can also be used with neural networks.
"""

import numpy as np
from typing import List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


class TfidfEmbedding:
    """TF-IDF based text embedding."""
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF embedding.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Tuple specifying n-gram range (e.g., (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term (as proportion)
            sublinear_tf: Apply sublinear TF scaling
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words='english'
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]) -> 'TfidfEmbedding':
        """
        Fit the TF-IDF vectorizer on texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Self for chaining
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Transform texts to TF-IDF vectors (sparse format).
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, max_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.transform(texts)
    
    def transform_dense(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to dense TF-IDF vectors.
        
        Args:
            texts: List of text documents
            
        Returns:
            Dense matrix of shape (n_samples, max_features)
        """
        sparse_mat = self.transform(texts)
        return sparse_mat.toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step (returns dense).
        
        Args:
            texts: List of text documents
            
        Returns:
            Dense matrix of shape (n_samples, max_features)
        """
        self.fit(texts)
        return self.transform_dense(texts)
    
    def fit_transform_sparse(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Fit vectorizer and transform texts (returns sparse).
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of shape (n_samples, max_features)
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_vocabulary(self) -> dict:
        """
        Get the vocabulary mapping words to feature indices.
        
        Returns:
            Dictionary mapping words to indices
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.vocabulary_
    
    def get_feature_names(self) -> List[str]:
        """
        Get all feature names (terms).
        
        Returns:
            List of term strings
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension (vocabulary size).
        
        Returns:
            Vocabulary size
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return len(self.vectorizer.vocabulary_)
    
    def get_top_terms(self, n: int = 20) -> List[str]:
        """
        Get top n terms in the vocabulary (by index).
        
        Args:
            n: Number of top terms to return
            
        Returns:
            List of top n term strings
        """
        features = self.get_feature_names()
        return features[:n]
    
    def get_document_sparsity(self, sparse_vectors: scipy.sparse.csr_matrix) -> float:
        """
        Calculate sparsity of document vectors.
        
        Args:
            sparse_vectors: Sparse matrix of vectors
            
        Returns:
            Sparsity percentage (0-100)
        """
        n_elements = sparse_vectors.shape[0] * sparse_vectors.shape[1]
        n_zeros = n_elements - sparse_vectors.nnz
        return (n_zeros / n_elements) * 100


class TfidfEmbeddingConfig:
    """Configuration template for TF-IDF embeddings."""
    
    @staticmethod
    def conservative() -> dict:
        """Conservative TF-IDF settings - fewer features, higher thresholds."""
        return {
            "max_features": 3000,
            "ngram_range": (1, 1),
            "min_df": 5,
            "max_df": 0.9,
            "sublinear_tf": True
        }
    
    @staticmethod
    def balanced() -> dict:
        """Balanced TF-IDF settings - moderate sparsity."""
        return {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True
        }
    
    @staticmethod
    def aggressive() -> dict:
        """Aggressive TF-IDF settings - more features, wider range."""
        return {
            "max_features": 10000,
            "ngram_range": (1, 3),
            "min_df": 1,
            "max_df": 0.99,
            "sublinear_tf": True
        }


def get_tfidf_embedding(config_name: str = "balanced") -> TfidfEmbedding:
    """
    Get a preconfigured TF-IDF embedding.
    
    Args:
        config_name: One of 'conservative', 'balanced', 'aggressive'
        
    Returns:
        Configured TfidfEmbedding instance
    """
    configs = {
        "conservative": TfidfEmbeddingConfig.conservative(),
        "balanced": TfidfEmbeddingConfig.balanced(),
        "aggressive": TfidfEmbeddingConfig.aggressive(),
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
    
    return TfidfEmbedding(**configs[config_name])
