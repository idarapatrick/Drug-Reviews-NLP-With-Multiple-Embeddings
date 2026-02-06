"""
Exploratory Data Analysis utilities for drug reviews dataset.

This module provides:
- Statistical summaries
- Distribution analysis
- Text statistics
- Visualization helpers
- Class balance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    """Comprehensive EDA toolkit for drug review data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with a DataFrame.
        
        Args:
            df: Input DataFrame
        """
        self.df = df
        self.text_column = None
        self.label_column = None
        
    def set_columns(self, text_column: str, label_column: str):
        """
        Specify which columns contain text and labels.
        
        Args:
            text_column: Name of text column
            label_column: Name of label/target column
        """
        self.text_column = text_column
        self.label_column = label_column
    
    def basic_info(self) -> Dict:
        """
        Get basic dataset information.
        
        Returns:
            Dictionary with info
        """
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "duplicates": self.df.duplicated().sum(),
        }
    
    def missing_values_analysis(self) -> Dict:
        """
        Analyze missing values.
        
        Returns:
            Dictionary with missing value stats
        """
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        return pd.DataFrame({
            "missing_count": missing,
            "missing_percentage": missing_percent
        }).to_dict()
    
    def text_statistics(self) -> Dict:
        """
        Get statistics about text data.
        
        Returns:
            Dictionary with text statistics
        """
        if self.text_column is None:
            return {"error": "Text column not set. Use set_columns() first."}
        
        texts = self.df[self.text_column].dropna()
        
        # Calculate statistics
        char_counts = texts.str.len()
        word_counts = texts.str.split().str.len()
        sentence_counts = texts.str.count(r'[.!?]+')
        
        return {
            "total_texts": len(texts),
            "avg_characters": char_counts.mean(),
            "min_characters": char_counts.min(),
            "max_characters": char_counts.max(),
            "std_characters": char_counts.std(),
            "avg_words": word_counts.mean(),
            "min_words": word_counts.min(),
            "max_words": word_counts.max(),
            "std_words": word_counts.std(),
            "avg_sentences": sentence_counts.mean(),
            "vocabulary_size": len(set(' '.join(texts).split())),
        }
    
    def label_distribution(self) -> Dict:
        """
        Analyze label distribution.
        
        Returns:
            Dictionary with class distribution
        """
        if self.label_column is None:
            return {"error": "Label column not set. Use set_columns() first."}
        
        dist = self.df[self.label_column].value_counts().to_dict()
        counts = self.df[self.label_column].value_counts()
        total = len(self.df)
        
        result = {}
        for label, count in sorted(dist.items()):
            result[f"class_{label}"] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        return result
    
    def class_balance(self) -> Dict:
        """
        Check if dataset is balanced.
        
        Returns:
            Dictionary with balance metrics
        """
        if self.label_column is None:
            return {"error": "Label column not set. Use set_columns() first."}
        
        counts = self.df[self.label_column].value_counts()
        max_count = counts.max()
        min_count = counts.min()
        
        return {
            "imbalance_ratio": max_count / min_count,
            "most_common_class": counts.idxmax(),
            "least_common_class": counts.idxmin(),
            "is_balanced": max_count / min_count < 1.5,
        }
    
    def numerical_summary(self) -> Dict:
        """
        Get summary statistics for numerical columns.
        
        Returns:
            Dictionary with numerical stats
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numeric_cols].describe().to_dict()
    
    def categorical_summary(self) -> Dict:
        """
        Get summary for categorical columns.
        
        Returns:
            Dictionary with categorical stats
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        summary = {}
        
        for col in categorical_cols:
            summary[col] = {
                "unique_values": self.df[col].nunique(),
                "top_value": self.df[col].value_counts().index[0] if len(self.df[col].value_counts()) > 0 else None,
                "top_value_count": self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0,
            }
        
        return summary
    
    def get_word_frequency(self, top_n: int = 20) -> Dict:
        """
        Get most common words in the dataset.
        
        Args:
            top_n: Number of top words to return
            
        Returns:
            Dictionary with word frequencies
        """
        if self.text_column is None:
            return {"error": "Text column not set. Use set_columns() first."}
        
        texts = self.df[self.text_column].dropna()
        words = ' '.join(texts).lower().split()
        word_freq = Counter(words)
        
        return dict(word_freq.most_common(top_n))
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            String with formatted report
        """
        report = "=" * 80 + "\n"
        report += "EXPLORATORY DATA ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Basic Info
        info = self.basic_info()
        report += "BASIC INFORMATION\n"
        report += f"  Shape: {info['shape']}\n"
        report += f"  Columns: {', '.join(info['columns'])}\n"
        report += f"  Memory Usage: {info['memory_usage_mb']:.2f} MB\n"
        report += f"  Duplicates: {info['duplicates']}\n\n"
        
        # Missing Values
        report += "MISSING VALUES\n"
        missing = self.missing_values_analysis()
        for col, stats in missing['missing_count'].items():
            if stats > 0:
                pct = missing['missing_percentage'][col]
                report += f"  {col}: {stats} ({pct:.2f}%)\n"
        report += "\n"
        
        # Text Statistics
        if self.text_column:
            report += "TEXT STATISTICS\n"
            text_stats = self.text_statistics()
            for key, value in text_stats.items():
                if isinstance(value, float):
                    report += f"  {key}: {value:.2f}\n"
                else:
                    report += f"  {key}: {value}\n"
            report += "\n"
        
        # Label Distribution
        if self.label_column:
            report += "LABEL DISTRIBUTION\n"
            dist = self.label_distribution()
            for label, info in dist.items():
                report += f"  {label}: {info['count']} ({info['percentage']:.1f}%)\n"
            report += "\n"
            
            # Class Balance
            balance = self.class_balance()
            report += "CLASS BALANCE\n"
            report += f"  Imbalance Ratio: {balance['imbalance_ratio']:.2f}\n"
            report += f"  Is Balanced: {balance['is_balanced']}\n"
            report += "\n"
        
        # Word Frequency
        if self.text_column:
            report += "TOP 10 WORDS\n"
            word_freq = self.get_word_frequency(top_n=10)
            for word, freq in word_freq.items():
                report += f"  {word}: {freq}\n"
        
        report += "\n" + "=" * 80
        
        return report
    
    def plot_label_distribution(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot label distribution.
        
        Args:
            figsize: Figure size
        """
        if self.label_column is None:
            print("Label column not set. Use set_columns() first.")
            return
        
        plt.figure(figsize=figsize)
        self.df[self.label_column].value_counts().sort_index().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot distribution of text lengths.
        
        Args:
            figsize: Figure size
        """
        if self.text_column is None:
            print("Text column not set. Use set_columns() first.")
            return
        
        texts = self.df[self.text_column].dropna()
        text_lengths = texts.str.split().str.len()
        
        plt.figure(figsize=figsize)
        plt.hist(text_lengths, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Text Lengths (Word Count)')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    
    def plot_word_frequency(self, top_n: int = 15, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot top N most frequent words.
        
        Args:
            top_n: Number of top words to plot
            figsize: Figure size
        """
        if self.text_column is None:
            print("Text column not set. Use set_columns() first.")
            return
        
        word_freq = self.get_word_frequency(top_n=top_n)
        words = list(word_freq.keys())
        freqs = list(word_freq.values())
        
        plt.figure(figsize=figsize)
        plt.barh(words, freqs)
        plt.xlabel('Frequency')
        plt.title(f'Top {top_n} Most Frequent Words')
        plt.tight_layout()
        plt.show()
