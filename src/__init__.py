"""
Shared NLP preprocessing and EDA module for drug reviews analysis.

This module provides utilities for:
- Loading and managing data
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
"""

__version__ = "1.0.0"

# Optional imports - only import if modules exist
__all__ = []

try:
    from .data_utils import DataLoader, DrugReviewDataset
    __all__.extend(["DataLoader", "DrugReviewDataset"])
except ImportError:
    pass

try:
    from .eda import EDAAnalyzer
    __all__.append("EDAAnalyzer")
except ImportError:
    pass

try:
    from .preprocessing import TextPreprocessor
    __all__.append("TextPreprocessor")
except ImportError:
    pass
