"""
Text preprocessing utilities for NLP tasks.

This module provides:
- Text cleaning and normalization
- Tokenization
- Stop word removal
- Lemmatization/Stemming
- Custom preprocessing pipelines
"""

import re
import string
from typing import List, Callable, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_special_chars: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        min_length: int = 1
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            remove_special_chars: Remove special characters
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming (usually used instead of lemmatize)
            min_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_length = min_length
        
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def clean_text(self, text: str) -> str:
        """
        Apply all configured cleaning operations to text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Handle null values
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters (but keep spaces and basic punctuation for now)
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;]', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            tokens = [t for t in tokens if t not in string.punctuation]
        
        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_length]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Stem
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def process(self, text: str) -> str:
        """
        Complete preprocessing: clean, tokenize, and rejoin.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of processed texts
        """
        return [self.process(text) for text in texts]
    
    def get_tokens_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Get tokenized representation of multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of token lists
        """
        return [self.tokenize(self.clean_text(text)) for text in texts]


class AdvancedTextPreprocessor(TextPreprocessor):
    """Extended preprocessor with additional features."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as TextPreprocessor."""
        super().__init__(*args, **kwargs)
        self.vocabulary = None
        
    def get_sentence_list(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 1) -> dict:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts to build vocabulary from
            min_freq: Minimum frequency for a word to be included
            
        Returns:
            Vocabulary dictionary mapping words to indices
        """
        word_freq = {}
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        filtered_vocab = {w: f for w, f in word_freq.items() if f >= min_freq}
        
        # Create word-to-index mapping (0 is reserved for padding)
        self.vocabulary = {word: idx + 1 for idx, word in enumerate(sorted(filtered_vocab.keys()))}
        self.vocabulary['<PAD>'] = 0
        
        return self.vocabulary
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of word indices.
        
        Args:
            texts: List of texts
            
        Returns:
            List of integer sequences
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        sequences = []
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            sequence = [self.vocabulary.get(token, self.vocabulary.get('<UNK>', 0)) for token in tokens]
            sequences.append(sequence)
        
        return sequences


# Predefined preprocessing configurations
MINIMAL_PREPROCESSING = {
    "lowercase": True,
    "remove_urls": True,
    "remove_html": True,
}

MODERATE_PREPROCESSING = {
    "lowercase": True,
    "remove_urls": True,
    "remove_html": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
}

AGGRESSIVE_PREPROCESSING = {
    "lowercase": True,
    "remove_urls": True,
    "remove_html": True,
    "remove_punctuation": True,
    "remove_numbers": True,
    "remove_stopwords": True,
    "lemmatize": True,
    "min_length": 2,
}


def get_preprocessor(config_name: str = "moderate") -> TextPreprocessor:
    """
    Get a preconfigured preprocessor.
    
    Args:
        config_name: One of 'minimal', 'moderate', 'aggressive'
        
    Returns:
        Configured TextPreprocessor
    """
    configs = {
        "minimal": MINIMAL_PREPROCESSING,
        "moderate": MODERATE_PREPROCESSING,
        "aggressive": AGGRESSIVE_PREPROCESSING,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
    
    return TextPreprocessor(**configs[config_name])
