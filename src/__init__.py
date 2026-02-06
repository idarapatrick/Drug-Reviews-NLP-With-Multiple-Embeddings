"""
Shared NLP preprocessing and EDA module for drug reviews analysis.

This module provides utilities for:
- Loading and managing data
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
"""

from .data_utils import DataLoader, DrugReviewDataset
from .eda import EDAAnalyzer
from .preprocessing import TextPreprocessor

__version__ = "1.0.0"
__all__ = ["DataLoader", "DrugReviewDataset", "EDAAnalyzer", "TextPreprocessor"]
