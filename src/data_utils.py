"""
Data loading and management utilities for drug reviews dataset.

This module handles:
- Loading CSV data
- Splitting datasets
- Creating data generators
- Managing train/validation/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and manage data from CSV files."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file containing drug reviews
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame with loaded data
        """
        if self.data is None:
            self.data = pd.read_csv(self.data_path)
        return self.data
    
    def get_info(self) -> Dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary with dataset info
        """
        if self.data is None:
            self.load_data()
            
        return {
            "num_samples": len(self.data),
            "num_features": len(self.data.columns),
            "columns": list(self.data.columns),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "missing_values": self.data.isnull().sum().to_dict(),
        }
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set from remaining data
            random_state: Random seed
            stratify_col: Column name to stratify by (for balanced splits)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            self.load_data()
        
        # Split into train+val and test
        stratify = self.data[stratify_col] if stratify_col else None
        train_val, test = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Split train+val into train and val
        stratify_tv = train_val[stratify_col] if stratify_col else None
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_tv
        )
        
        return train, val, test
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get random samples from the dataset.
        
        Args:
            n: Number of samples to return
            
        Returns:
            DataFrame with n random samples
        """
        if self.data is None:
            self.load_data()
        return self.data.sample(n=min(n, len(self.data)))


class DrugReviewDataset:
    """Wrapper class for drug review dataset with preprocessing support."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        ratings: Optional[List[float]] = None,
        drug_names: Optional[List[str]] = None
    ):
        """
        Initialize drug review dataset.
        
        Args:
            texts: List of review texts
            labels: List of binary/multi-class labels
            ratings: Optional list of ratings
            drug_names: Optional list of drug names
        """
        self.texts = texts
        self.labels = np.array(labels)
        self.ratings = np.array(ratings) if ratings else None
        self.drug_names = drug_names
        self.processed_texts = None
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get single data point.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with text and label
        """
        text = self.processed_texts[idx] if self.processed_texts is not None else self.texts[idx]
        return {
            "text": text,
            "label": self.labels[idx],
            "rating": self.ratings[idx] if self.ratings is not None else None,
            "drug_name": self.drug_names[idx] if self.drug_names else None
        }
    
    def set_processed_texts(self, texts: List[str]):
        """
        Set preprocessed texts.
        
        Args:
            texts: List of preprocessed texts
        """
        assert len(texts) == len(self.texts), "Processed texts must match original length"
        self.processed_texts = texts
    
    def get_unprocessed(self) -> List[str]:
        """Return original unprocessed texts."""
        return self.texts
    
    def get_processed(self) -> List[str]:
        """Return processed texts."""
        if self.processed_texts is None:
            return self.texts
        return self.processed_texts
    
    def get_labels(self) -> np.ndarray:
        """Return labels."""
        return self.labels
    
    def class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    rating_column: Optional[str] = None,
    drug_column: Optional[str] = None
) -> DrugReviewDataset:
    """
    Create DrugReviewDataset from pandas DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing review texts
        label_column: Name of column containing labels
        rating_column: Optional column name for ratings
        drug_column: Optional column name for drug names
        
    Returns:
        DrugReviewDataset instance
    """
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    ratings = df[rating_column].tolist() if rating_column and rating_column in df.columns else None
    drugs = df[drug_column].tolist() if drug_column and drug_column in df.columns else None
    
    return DrugReviewDataset(texts, labels, ratings, drugs)
