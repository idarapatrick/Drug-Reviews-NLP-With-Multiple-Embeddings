# Quick Start Guide for Team Members

## What's Been Created

The repository now has a complete shared infrastructure for the drug reviews NLP project:

### 1. **Shared Modules** (`src/` folder)
- **`data_utils.py`**: Load, explore, and split data
- **`preprocessing.py`**: Tokenize, clean, and preprocess text
- **`eda.py`**: Generate reports and visualizations

### 2. **Shared Embeddings** (`embeddings/` folder)
- **`tfidf_embedding.py`**: TF-IDF vectorization (traditional ML)
- **`word2vec_embedding.py`**: Word2Vec Skip-gram (SELECTED)
- **`glove_embedding.py`**: GloVe word embeddings
- All with consistent interface for easy swapping

### 3. **Example Notebooks** (`notebooks/` folder)
- **`0_shared_eda_and_preprocessing.ipynb`**: Complete guide for using shared utilities
- **`1_gru_template.ipynb`**: Template for GRU implementation (your model!)

---

## How to Use This for Your Work

### Step 1: Run the Shared EDA Notebook
1. Open `notebooks/0_shared_eda_and_preprocessing.ipynb`
2. Ensure the DATA_PATH points to the tsv(tab seperated values) file
3. Run all cells - this creates preprocessed datasets ready for embeddings

### Step 2: Create Your Model Notebook (All 3 Embeddings in ONE file)
For your assigned model (GRU/RNN/LSTM/Transformer), create ONE notebook that implements your model with all 3 team-selected embeddings:

**Example: gru_model.ipynb Structure**
```
Section 1: Data Loading & Preprocessing (run once)
Section 2: Build & Train GRU with Word2Vec Skip-gram
Section 3: Build & Train GRU with GloVe
Section 4: Build & Train GRU with TF-IDF
Section 5: Compare Results
```

This structure allows you to:
- Share data loading and preprocessing code
- Build your model architecture 3 times (once per embedding)
- Compare performance across embeddings in one place
- Track which embedding works best for your model

**Set configurations for each embedding section**:
```python
# Section 2: Word2Vec
embedding = get_embedding('word2vec', 'skipgram_medium')
token_based = True  # requires tokenization

# Section 3: GloVe
embedding = get_embedding('glove', 'medium')
token_based = True  # requires tokenization

# Section 4: TF-IDF
embedding = get_embedding('tfidf', 'balanced')
token_based = False  # uses raw text
```

3. Run the entire notebook and observe your results

### Step 3: Document Your Results
Save your results in a dictionary/JSON format for later synthesis:
```python
results = {
    "model": "GRU",
    "embedding": "word2vec_skipgram",
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.88,
    "f1_score": 0.86
}
```

---

## For Other Team Members (RNN, LSTM, Transformer)

The same approach works for your models:
1. Create your model template in `notebooks/`
2. Use the shared embeddings and preprocessing
3. Swap only the model architecture

**Key Point**: You don't rewrite preprocessing or embedding code - you import and use the shared modules!

---

## Important Configuration Variables

### In `0_shared_eda_and_preprocessing.ipynb`:
```python
DATA_PATH = '../data/drugLibTrain_raw.tsv'  # TSV file
TEXT_COLUMN = 'commentsReview'  # Main review text column
LABEL_COLUMN = 'rating'  # Target variable
FILE_FORMAT = 'tsv'  # Tab-separated values
PREPROCESSING_CONFIG = 'moderate'  # 'minimal', 'moderate', or 'aggressive'
```

### In your model notebooks (finalized team embeddings):
```python
# Use these three embeddings for your GRU models:
EMBEDDING_TYPE = 'word2vec'  # skipgram_medium (primary)
EMBEDDING_CONFIG = 'skipgram_medium'

# OR
EMBEDDING_TYPE = 'glove'  # medium (secondary)
EMBEDDING_CONFIG = 'medium'

# OR
EMBEDDING_TYPE = 'tfidf'  # balanced (baseline)
EMBEDDING_CONFIG = 'balanced'

BATCH_SIZE = 32
EPOCHS = 10
GRU_UNITS = 128
```

---

## Next Steps for Team

### Immediate Actions:
1. [DONE] Create 3 shared embeddings (Done: Word2Vec Skip-gram, GloVe, TF-IDF)
2. [IN PROGRESS] Each member creates ONE notebook with their model and all 3 embeddings
3. [PENDING] Run all models and collect results
4. [PENDING] Create comparison tables and visualizations

### Final Embeddings Chosen (USE ALL 3):
- **Word2Vec Skip-gram** (skipgram_medium) - Primary semantic embedding
- **GloVe** (medium) - Global context embedding
- **TF-IDF** (balanced) - Statistical baseline embedding

### Expected Notebooks:
- **Idara**: `gru_model.ipynb` (GRU with all 3 embeddings)
- **Patrick**: `rnn_model.ipynb` (RNN with all 3 embeddings)
- **Gershom**: `lstm_model.ipynb` (LSTM with all 3 embeddings)
- **Elissa**: `transformer_model.ipynb` (Transformer with all 3 embeddings)
5. Write the academic report

---

## Troubleshooting

### ImportError: No module named 'src'
Make sure you're running from inside the `notebooks/` folder and have `sys.path.append('../')` at the top

### Data file not found
Ensure `drugLibTrain_raw.tsv` and `drugLibTest_raw.tsv` are in the `data/` folder

### NLTK data missing
Run in a cell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---
 
**Group Project**: Comparative Analysis of Text Classification with Multiple Embeddings
