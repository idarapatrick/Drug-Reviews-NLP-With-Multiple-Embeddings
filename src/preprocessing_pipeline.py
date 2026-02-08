"""
Preprocessing Pipeline for Drug Review Sentiment Analysis

This module provides standardized data loading and preprocessing for training GRU models
with different embedding approaches. It handles train, validation, and test datasets
consistently and prepares both sequence data and TF-IDF representations.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# Global Configuration
MAX_LEN = 100
MAX_WORDS = 20000
TFIDF_FEATURES = 2000


# Preprocessing class for LSTM notebooks
class TextPreprocessor:
    """
    Text preprocessing for tokenization and cleaning.
    Compatible with Word2Vec, GloVe, and TF-IDF embeddings.
    """
    def __init__(
        self,
        lowercase=True,
        remove_stopwords=False,
        lemmatize=False,
        remove_punctuation=False
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Preprocess a single text string."""
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_tokens(self, text):
        """Convert text to list of tokens."""
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            tokens = [t for t in tokens if t.isalnum()]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def get_tokens_batch(self, texts):
        """Convert list of texts to list of token lists."""
        return [self.get_tokens(text) for text in texts]
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts."""
        return [self.preprocess_text(text) for text in texts]


def get_preprocessor(config='moderate'):
    """
    Get a text preprocessor with predefined configuration.
    
    Args:
        config: 'minimal', 'moderate', or 'aggressive'
    
    Returns:
        TextPreprocessor instance
    """
    configs = {
        'minimal': {
            'lowercase': True,
            'remove_stopwords': False,
            'lemmatize': False,
            'remove_punctuation': False
        },
        'moderate': {
            'lowercase': True,
            'remove_stopwords': True,
            'lemmatize': False,
            'remove_punctuation': False
        },
        'aggressive': {
            'lowercase': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'remove_punctuation': True
        }
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")
    
    return TextPreprocessor(**configs[config])


def load_and_clean_csv(filepath):
    """
    Load a CSV file and apply sentiment classification rules.
    
    Reviews are classified as positive (rating 7-10) or negative (rating 1-4).
    Neutral reviews (rating 5-6) are excluded from the dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with 'sentiment' column (0 for negative, 1 for positive)
    """
    df = pd.read_csv(filepath, index_col=0)
    
    # Filter: keep only positive (7-10) and negative (1-4) reviews
    df = df[(df['rating'] <= 4) | (df['rating'] >= 7)].copy()
    
    # Create binary sentiment: 1 for positive, 0 for negative
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 7 else 0)
    
    # Clean review text
    df['review'] = df['review'].astype(str).fillna('')
    
    return df


def prepare_datasets(train_path=None, val_path=None, test_path=None):
    """
    Load and preprocess train, validation, and test datasets.
    
    This function fits the tokenizer and TF-IDF vectorizer on the training set
    and transforms all three datasets using these fitted objects to prevent
    data leakage from validation or test sets.
    
    Args:
        train_path: Path to training CSV (default: 'data/drug_review_train.csv')
        val_path: Path to validation CSV (default: 'data/drug_review_validation.csv')
        test_path: Path to test CSV (default: 'data/drug_review_test.csv')
        
    Returns:
        Dictionary containing:
            - train: (X_seq, X_tfidf, y) for training set
            - val: (X_seq, X_tfidf, y) for validation set
            - test: (X_seq, X_tfidf, y) for test set
            - vocab_size: Vocabulary size from tokenizer
            - tokenizer: Fitted tokenizer object
            - tfidf: Fitted TF-IDF vectorizer object
    """
    # Use default paths if not provided
    if train_path is None:
        train_path = 'data/drug_review_train.csv'
    if val_path is None:
        val_path = 'data/drug_review_validation.csv'
    if test_path is None:
        test_path = 'data/drug_review_test.csv'
    
    print("Loading training data")
    df_train = load_and_clean_csv(train_path)
    print(f"Training set: {df_train.shape[0]} samples")
    
    print("Loading validation data")
    df_val = load_and_clean_csv(val_path)
    print(f"Validation set: {df_val.shape[0]} samples")
    
    print("Loading test data")
    df_test = load_and_clean_csv(test_path)
    print(f"Test set: {df_test.shape[0]} samples")
    
    # Fit tokenizer on training data only
    print("\nTokenizing sequences")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(df_train['review'])
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")
    
    # Transform all datasets
    train_sequences = tokenizer.texts_to_sequences(df_train['review'])
    val_sequences = tokenizer.texts_to_sequences(df_val['review'])
    test_sequences = tokenizer.texts_to_sequences(df_test['review'])
    
    X_seq_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_seq_val = pad_sequences(val_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_seq_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Fit TF-IDF on training data only
    tfidf = TfidfVectorizer(max_features=TFIDF_FEATURES, stop_words='english')
    X_tfidf_train = tfidf.fit_transform(df_train['review']).toarray()
    X_tfidf_val = tfidf.transform(df_val['review']).toarray()
    X_tfidf_test = tfidf.transform(df_test['review']).toarray()
    
    # Extract labels
    y_train = df_train['sentiment'].values
    y_val = df_val['sentiment'].values
    y_test = df_test['sentiment'].values
    
    print(f"\nDataset shapes:")
    print(f"  Train: X_seq={X_seq_train.shape}, X_tfidf={X_tfidf_train.shape}, y={y_train.shape}")
    print(f"  Val:   X_seq={X_seq_val.shape}, X_tfidf={X_tfidf_val.shape}, y={y_val.shape}")
    print(f"  Test:  X_seq={X_seq_test.shape}, X_tfidf={X_tfidf_test.shape}, y={y_test.shape}")
    
    print(f"\nSentiment distribution:")
    print(f"  Train: Negative={np.sum(y_train==0)}, Positive={np.sum(y_train==1)}")
    print(f"  Val:   Negative={np.sum(y_val==0)}, Positive={np.sum(y_val==1)}")
    print(f"  Test:  Negative={np.sum(y_test==0)}, Positive={np.sum(y_test==1)}")
    
    return {
        'train': (X_seq_train, X_tfidf_train, y_train),
        'val': (X_seq_val, X_tfidf_val, y_val),
        'test': (X_seq_test, X_tfidf_test, y_test),
        'vocab_size': vocab_size,
        'tokenizer': tokenizer,
        'tfidf': tfidf
    }


if __name__ == "__main__":
    data = prepare_datasets()
