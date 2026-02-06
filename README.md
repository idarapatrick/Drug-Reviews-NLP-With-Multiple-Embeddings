# Drug Reviews NLP - Comparative Analysis of Text Classification with Multiple Embeddings

## Project Overview

This is a group project comparing text classification performance using different deep learning models (RNN, LSTM, GRU, Transformer) with team-selected embedding techniques.

**Team Focus:**
- **Traditional ML Model**: Logistic Regression / SVM / Random Forest (to be assigned)
- **RNN Models**: [Team member name]
- **LSTM Models**: [Team member name]
- **GRU Models**: Essie (this repository)
- **Transformer Models**: [Team member name]

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                              # Data directory
│   └── README.md                      # Data documentation
├── src/                               # Shared utility modules
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading and management
│   ├── preprocessing.py               # Text preprocessing
│   └── eda.py                         # Exploratory data analysis
├── embeddings/                        # Embedding implementations (SHARED)
│   ├── tfidf_embedding.py             # TF-IDF vectorization
│   ├── word2vec_embedding.py          # Word2Vec (Skip-gram SELECTED)
│   └── glove_embedding.py             # GloVe embedding (SELECTED)
├── notebooks/                         # Individual team member notebooks
│   ├── 0_shared_eda_and_preprocessing.ipynb   # Shared preprocessing pipeline
│   ├── 1_gru_template.ipynb                   # GRU template (select embedding)
│   ├── gru_word2vec_skipgram.ipynb    # GRU with Word2Vec Skip-gram (PRIMARY)
│   ├── gru_glove.ipynb                # GRU with GloVe (SECONDARY)
│   └── gru_tfidf.ipynb                # GRU with TF-IDF (BASELINE)
└── config/                            # Configuration files
    └── hyperparameters.yaml           # Model hyperparameters
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Load and Explore Data

```python
import pandas as pd
from src.eda import EDAAnalyzer

# Load TSV data
df = pd.read_csv('data/drugLibTrain_raw.tsv', sep='\t')

print(f"Dataset size: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Columns: {list(df.columns)}")

# Explore data
analyzer = EDAAnalyzer(df)
analyzer.set_columns(text_column='commentsReview', label_column='rating')
print(analyzer.generate_report())

# Visualize
analyzer.plot_label_distribution()
analyzer.plot_text_length_distribution()
```

### 3. Preprocess Text

```python
from src.preprocessing import TextPreprocessor, get_preprocessor

# Use a predefined configuration
preprocessor = get_preprocessor("moderate")  # Options: 'minimal', 'moderate', 'aggressive'

# Or create custom preprocessing
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_stopwords=True,
    lemmatize=True
)

# Process single text
cleaned = preprocessor.process("This is a test review!")

# Process batch
texts = ["Review 1", "Review 2"]
processed = preprocessor.process_batch(texts)
```

### 4. Create Dataset Object

```python
from src.data_utils import create_dataset_from_dataframe

dataset = create_dataset_from_dataframe(
    df=df,
    text_column='commentsReview',
    label_column='rating',
    rating_column='rating',
    drug_column='urlDrugName'
)

# Get processed dataset
preprocessor = get_preprocessor("moderate")
processed_texts = preprocessor.process_batch(dataset.get_unprocessed())
dataset.set_processed_texts(processed_texts)
```

## Shared Modules

### Data Utilities (`src/data_utils.py`)

**DataLoader**
- `load_data()`: Load CSV files
- `get_info()`: Get dataset statistics
- `split_data()`: Train/val/test split with stratification
- `get_sample()`: Get random samples

**DrugReviewDataset**
- Custom dataset class with managing processed/unprocessed text
- Support for labels, ratings, and drug names
- Easy integration with PyTorch/TensorFlow models

### Text Preprocessing (`src/preprocessing.py`)

**TextPreprocessor**
- Configurable cleaning (URLs, HTML, special chars, numbers)
- Tokenization with various options
- Stopword removal
- Lemmatization and stemming
- Built-in NLTK integration

**AdvancedTextPreprocessor**
- Vocabulary building
- Sequence conversion for embeddings
- Sentence segmentation

**Predefined Configurations**
- `minimal`: Basic cleaning
- `moderate`: Cleaning + stopword removal
- `aggressive`: Maximum cleaning with lemmatization

### Exploratory Data Analysis (`src/eda.py`)

**EDAAnalyzer**
- `basic_info()`: Dataset dimensions and metadata
- `missing_values_analysis()`: Missing data stats
- `text_statistics()`: Text length, word count, vocabulary size
- `label_distribution()`: Class distribution analysis
- `class_balance()`: Imbalance detection
- `get_word_frequency()`: Most common words
- `generate_report()`: Comprehensive text report
- Visualization methods for distributions and word frequency

## Embedding Implementations (Shared)

Each embedding module follows a consistent interface for easy integration:

```python
from embeddings.word2vec_embedding import Word2VecEmbedding

# Initialize embedding
embed = Word2VecEmbedding(embedding_dim=100)

# Train on texts
embed.train(texts)

# Get embeddings
word_vectors = embed.get_word_vector('drug')
sentence_vectors = embed.encode_texts(texts)
```

### Available Embeddings
- **TF-IDF**: `embeddings/tfidf_embedding.py`
- **Word2Vec (Skip-gram SELECTED)**: `embeddings/word2vec_embedding.py`
- **GloVe**: `embeddings/glove_embedding.py`

## Model Implementation Workflow

### For Team Members Using Shared Code:

1. **Set up your notebook** in `notebooks/` folder:
   ```python
   import sys
   sys.path.append('../')
   from src.data_utils import DataLoader, create_dataset_from_dataframe
   from src.preprocessing import get_preprocessor
   from src.eda import EDAAnalyzer
   from embeddings.word2vec_embedding import Word2VecEmbedding
   ```

2. **Load and preprocess data** using shared utilities

3. **Create embeddings** using shared embedding classes

4. **Build your model** (RNN/LSTM/GRU/Transformer) with the embedded data

5. **Document results** in a results file for synthesis

## Finalized Embeddings (Team Decision)

**Selected for all models (GRU, RNN, LSTM, Transformer):**
1. [DONE] Word2Vec Skip-gram (skipgram_medium) - Primary semantic embedding
2. [DONE] GloVe (medium) - Global context embedding
3. [DONE] TF-IDF (balanced) - Statistical baseline embedding

**Why these three?**
- Represent different embedding philosophies (semantic vs. global vs. statistical)
- Word2Vec Skip-gram: Best for understanding drug/symptom/benefit relationships
- GloVe: Captures global patterns in medical language
- TF-IDF: Provides interpretable baseline for model comparison

**Next: Each member implements their model using all 3 embeddings**

## Documentation Standards

- Use docstrings for all functions and classes
- Include type hints
- Add usage examples in docstrings
- Update README when adding new utilities

## Git Workflow

```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Work on your contributions
# Commit with clear messages
git commit -m "Add GRU model with Word2Vec embeddings"

# Push and create PR
git push origin feature/your-feature
```

## Team Contribution Tracking

**Important**: Maintain a contribution tracker file to document:
- Who implemented what module
- Dates and hours spent
- Specific contributions to reproducibility

This is required for fair grading.

## References and Citations

Document all papers and resources used in:
- Code comments
- Docstring references
- Final report bibliography

## Troubleshooting

### NLTK Data Not Found
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Import Errors
Make sure you're running from the correct directory and the `src/` folder is in your Python path.

### Data Loading Issues
Verify the data path and column names match your CSV structure.

## Project Phases

1. **Phase 1** [COMPLETED]: Finalize embeddings (Word2Vec Skip-gram, GloVe, TF-IDF) and implement shared modules
2. **Phase 2** [IN PROGRESS]: Each member creates ONE notebook implementing their model with all 3 embeddings
3. **Phase 3** [PENDING]: Run experiments and collect cross-embedding results
4. **Phase 4** [PENDING]: Create comparison tables and visualizations
5. **Phase 5** [PENDING]: Write comprehensive academic report with results

---

**Last Updated**: February 6, 2026  
**Group Assignment**: Machine Learning Techniques I - Formative 2
