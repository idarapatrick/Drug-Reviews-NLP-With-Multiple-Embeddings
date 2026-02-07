# Data Directory

## Overview
This directory contains the drug reviews dataset used across all team member's experiments.

## Dataset Information

**Source**: UCI Machine Learning Repository - Drug Reviews Dataset  
**Task**: Text Classification (Rating Prediction)  
**Format**: TSV (Tab-Separated Values)
**Files**: drugLibTrain_raw.tsv, drugLibTest_raw.tsv

## Directory Structure

```
data/
├── README.md                    # This file
├── drugLibTrain_raw.tsv         # Main training dataset
├── drugLibTest_raw.tsv          # Test dataset
├── processed/                   # Processed versions (generated)
│   ├── train_processed.csv
│   ├── val_processed.csv
│   └── test_processed.csv
└── splits/                      # Different train/val/test splits
```

## Data Loading

### Basic Usage
```python
import pandas as pd
from src.preprocessing import get_preprocessor

# Load TSV file
df = pd.read_csv('data/drugLibTrain_raw.tsv', sep='\t')

# Get dataset info
info = loader.get_info()

# Split into train/val/test
train, val, test = loader.split_data(test_size=0.2, val_size=0.1)
```

## Column Descriptions

Columns in the drugLibTrain_raw.tsv / drugLibTest_raw.tsv:
- **urlDrugName**: Drug identifier/name
- **rating**: Target variable (numerical rating, typically 1-10)
- **effectiveness**: Effectiveness level
- **sideEffects**: Side effects severity
- **condition**: Medical condition being treated
- **benefitsReview**: Text describing benefits
- **sideEffectsReview**: Text describing side effects
- **commentsReview**: General comments/review text (main text column)

**Key Columns for This Project**:
- Text: `commentsReview`, `benefitsReview`, or `sideEffectsReview`
- Label: `rating` (target variable)
- Drug ID: `urlDrugName`

## Data Statistics

Once data is loaded, use EDAAnalyzer to get:
```python
from src.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df)
analyzer.set_columns(text_column='commentsReview', label_column='rating')
print(analyzer.generate_report())
```

## Preprocessing Notes

- All team members should use the shared preprocessing module
- Different embedding techniques may require different preprocessing levels
- Document preprocessing choices in your notebooks

## File Size and Memory

Monitor your system's available RAM when working with large datasets.

---

**Note**: Please do not commit large data files to git. Use .gitignore to exclude data files.
