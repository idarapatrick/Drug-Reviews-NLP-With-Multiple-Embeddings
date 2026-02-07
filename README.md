# Deep Learning for Drug Review Sentiment Analysis

This project evaluates GRU, LSTM, and Transformer architectures on the **UCI Drug Review Dataset (Drugs.com)** using a Hybrid Multi-Embedding approach.

## Dataset

- **Source**: `/data/drugsComTrain_raw.csv` (~161k rows after filtering)
- **Task**: Binary Classification
  - **Positive**: Rating >= 7
  - **Negative**: Rating <= 4
  - **Neutral (Dropped)**: Rating 5-6
- **Columns**: patient_id, drugName, condition, review, rating, date, usefulCount, review_length

## Architecture

We use a **Hybrid Multi-Embedding** approach combining sequence data with auxiliary features:

1. **Sequence Input (100 tokens)**: 
   - Tokenized reviews processed through Keras Tokenizer
   - Padded to fixed length (100)
   - Fed to Embedding layer + GRU/LSTM layers

2. **TF-IDF Input (2000 features)**:
   - Statistical text vectorization using sklearn
   - Fed to Dense layers in auxiliary branch
   - Concatenated with sequence output before final classification

3. **Output**: Binary classification (Positive/Negative)

## Repository Structure

```
.
├── README.md                          # This file
├── preprocessing_pipeline.py          # Source of truth for data preprocessing
├── EDA_standard.py                    # Exploratory data analysis
├── quick_setup.md                     # Quick setup guide
├── requirements.txt                   # Dependencies
├── data/                              # Data directory
│   ├── README.md
│   ├── drug_review_train.csv
│   ├── drug_review_validation.csv
│   └── drug_review_test.csv
├── src/                               # Shared utility modules
│   ├── __init__.py
│   ├── data_utils.py
│   ├── preprocessing.py
│   └── eda.py
├── embeddings/                        # Embedding implementations
│   ├── __init__.py
│   ├── tfidf_embedding.py
│   ├── word2vec_embedding.py
│   └── glove_embedding.py
├── notebooks/                         # Team member notebooks
│   ├── 0_shared_eda_and_preprocessing.ipynb
│   └── 1_gru_embeddings.ipynb
└── config/                            # Configuration
    ├── EMBEDDINGS_SELECTED.md
    └── hyperparameters.md
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Verify Data

Run the EDA to validate the data schema:

```bash
python EDA_standard.py
```

This generates:
- `sentiment_distribution.png`
- `review_length_distribution.png`

### 3. Load Data for Modeling

All team members must use `preprocessing_pipeline.py`:

```python
from preprocessing_pipeline import get_data_for_model

# Load and preprocess data
X_seq, X_tfidf, y, vocab_size, tokenizer, tfidf = get_data_for_model()

# X_seq: (N, 100) - Sequence input for GRU/LSTM
# X_tfidf: (N, 2000) - TF-IDF input for Dense layers
# y: (N,) - Binary sentiment labels
# vocab_size: Tokenizer vocabulary size
# tokenizer: Fitted Keras Tokenizer for inference
# tfidf: Fitted TfidfVectorizer for inference
```

## Model Architecture (Hybrid Sequence Model)

For Sequence Models (GRU/LSTM/Transformer implementations), follow this structure:

### Input Branch 1: Sequence Processing
```
Input(shape=(100,))
  ↓
Embedding(word2vec)
Embedding(glove)
  ↓
Concatenate embeddings
  ↓
GRU(64, return_sequences=False)
  ↓
[sequence_output shape: (batch_size, 64)]
```

### Input Branch 2: TF-IDF Processing
```
Input(shape=(2000,))
  ↓
Dense(32, relu)
  ↓
[tfidf_output shape: (batch_size, 32)]
```

### Merge & Output
```
Concatenate([sequence_output, tfidf_output])
  ↓
Dense(1, sigmoid)
  ↓
Binary output (Positive/Negative)
```

## Preprocessing Pipeline Details

### Target Generation
```python
rating >= 7 → 1 (Positive)
rating <= 4 → 0 (Negative)
rating 5 or 6 → Dropped (Neutral)
```

### Sequence Input
- **Tokenizer**: Keras Tokenizer with 20,000 max words
- **Sequences**: Convert text to token sequences
- **Padding**: `pad_sequences(maxlen=100, padding='post', truncating='post')`
- **Output Shape**: (Batch_Size, 100)

### Auxiliary Input
- **Vectorizer**: TF-IDF with 2,000 features
- **Stop words**: English stop words removed
- **Output Shape**: (Batch_Size, 2000)

## Configuration

### Global Parameters (in `preprocessing_pipeline.py`)
- `MAX_LEN`: 100 (sequence length)
- `MAX_WORDS`: 20,000 (tokenizer vocab size)
- `TFIDF_FEATURES`: 2,000 (TF-IDF dimensions)

### Hyperparameters (in notebook)
- GRU units: 64
- Dropout rate: 0.3
- Learning rate: Default Adam optimizer
- Batch size: 32
- Epochs: 20+

## Team Instructions

### For GRU/LSTM/Transformer Implementation:

1. **Import the pipeline**:
   ```python
   from preprocessing_pipeline import get_data_for_model
   X_seq, X_tfidf, y, vocab_size, tokenizer, tfidf = get_data_for_model()
   ```

2. **Build your model** with the Hybrid architecture above

3. **Use both inputs**:
   - Feed `X_seq` to the sequence branch (Embedding → GRU/LSTM)
   - Feed `X_tfidf` to the auxiliary branch (Dense layers)
   - Merge both outputs before final classification

4. **Train and evaluate** on test set

5. **Document results** comparing different embeddings

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

1. **Import the pipeline and build your model** with the Hybrid architecture

## References

- Dataset: UCI Drug Review Dataset (Drugs.com reviews)
- Text Processing: TensorFlow/Keras, scikit-learn
- Models: GRU/LSTM/Transformer attention mechanisms

---

**Last Updated**: February 7, 2026  
**Status**: Active Development - Hybrid Model Architecture
