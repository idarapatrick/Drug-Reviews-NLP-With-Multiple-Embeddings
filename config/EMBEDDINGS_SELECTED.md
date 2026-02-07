# Team-Finalized Embeddings Selection

## Date: February 6, 2026
## Decision: FINAL - Use across all models (GRU, RNN, LSTM, Transformer)

---

## Selected Embeddings

### 1. Word2Vec Skip-gram (PRIMARY)
**Status**: Selected as primary embedding  
**Config**: `skipgram_medium`  
**Why this embedding?**
- Best captures semantic relationships in drug review language
- Learns associations like: "drug_name" ↔ "side effect", "benefits" ↔ positive sentiment
- Excellent for small-to-medium datasets
- Works well with neural network architectures (GRU, LSTM, RNN, Transformer)

**Use in notebooks**:
```python
EMBEDDING_TYPE = 'word2vec'
EMBEDDING_CONFIG = 'skipgram_medium'
```

**Expected notebook**: `gru_word2vec_skipgram.ipynb`

---

### 2. GloVe (SECONDARY)
**Status**: Selected as secondary embedding  
**Config**: `medium`  
**Why this embedding?**
- Combines local context window with global corpus statistics
- Better captures pharmaceutical/medical terminology patterns
- Good for learning domain-specific relationships
- Complements Word2Vec for comparison and ensemble approaches

**Use in notebooks**:
```python
EMBEDDING_TYPE = 'glove'
EMBEDDING_CONFIG = 'medium'
```

**Expected notebook**: `gru_glove.ipynb`

---

### 3. TF-IDF (BASELINE)
**Status**: Selected as baseline/interpretable embedding  
**Config**: `balanced`  
**Why this embedding?**
- Provides interpretable, explainable feature importance rankings
- Useful for traditional machine learning models (Logistic Regression, SVM)
- Serves as comparison baseline to evaluate neural embeddings
- Fast and lightweight, no training required beyond vectorization

**Use in notebooks**:
```python
EMBEDDING_TYPE = 'tfidf'
EMBEDDING_CONFIG = 'balanced'
```

**Expected notebook**: `gru_tfidf.ipynb`

---

## Why NOT Selected?

### Word2Vec CBOW
- **Reason**: Skip-gram outperforms CBOW on smaller datasets and is more widely used for NLP
- **Redundancy**: Including both would not add meaningful comparison value
- **Decision**: Skip-gram alone is sufficient; CBOW excluded to focus efforts

### FastText
- **Reason**: Not implemented in current codebase; would require additional setup
- **Alternative**: Word2Vec handles OOV words adequately for this dataset

### BERT / Large Language Models
- **Reason**: Too computationally heavy for group project constraints
- **Scope**: Beyond scope; static embeddings sufficient for text classification task

---

## Implementation Checklist

### Phase 1: Setup (COMPLETED)
- [x] Create shared embedding modules (tfidf, word2vec, glove)
- [x] Create EDA and preprocessing modules
- [x] Test embeddings with sample data
- [x] Finalize embedding selection
- [x] Update documentation and notebooks

### Phase 2: Model Implementation (IN PROGRESS)
- [ ] Essie: Create `gru_word2vec_skipgram.ipynb` (Word2Vec Skip-gram)
- [ ] Essie: Create `gru_glove.ipynb` (GloVe)
- [ ] Essie: Create `gru_tfidf.ipynb` (TF-IDF)
- [ ] Other members: Create RNN/LSTM/Transformer variants with same 3 embeddings

### Phase 3: Experimentation (PENDING)
- [ ] Run all models with all 3 embeddings
- [ ] Document hyperparameters and random seeds
- [ ] Record accuracy, precision, recall, F1-scores
- [ ] Save model artifacts for reproducibility

### Phase 4: Analysis (PENDING)
- [ ] Create comparison tables (embedding × model)
- [ ] Generate visualizations (performance heatmaps, loss curves)
- [ ] Analyze which embedding-model combinations perform best
- [ ] Document domain-specific insights from embeddings

### Phase 5: Reporting (PENDING)
- [ ] Write academic report synthesizing all results
- [ ] Include discussion of embedding trade-offs
- [ ] Cite relevant embedding papers and methodologies
- [ ] Create contribution tracker for fair grading

---

## Key Configurations Across Project

### Standard Preprocessing (All Embeddings)
```python
PREPROCESSING_CONFIG = 'moderate'
# Includes: lowercase, URL removal, HTML removal, punctuation removal, stopword removal
# Does NOT include: number removal, lemmatization (too aggressive)
```

### Data Paths (All Notebooks)
```python
DATA_PATH = '../data/drugLibTrain_raw.tsv'
TEXT_COLUMN = 'commentsReview'
LABEL_COLUMN = 'rating'
FILE_FORMAT = 'tsv'  # Tab-separated values
```

### Model Hyperparameters (GRU - Adjust for other models)
```python
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
GRU_UNITS = 128
DROPOUT_RATE = 0.3
RANDOM_SEED = 42
```

---

## Rationale for Final Selection

**Why these 3?**
1. **Complement each other**: Cover neural semantic (Skip-gram), neural global (GloVe), and statistical (TF-IDF)
2. **Well-established**: All have proven track records in NLP classification tasks
3. **Representative**: Show different embedding philosophies for comprehensive comparison
4. **Balanced scope**: Not too many (keeps experiments manageable), not too few (provides meaningful comparison)
5. **Drug reviews domain**: All three handle medical/pharmaceutical terminology effectively

**Domain Fit:**
- Drug reviews have specialized language (drug names, dosages, side effects, efficacy ratings)
- Semantic understanding (Word2Vec) crucial for relating symptoms to drugs
- Global patterns (GloVe) help identify common phrases across reviews
- Statistical rarity (TF-IDF) can highlight distinctive side effects or benefits

---

## References

- Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space" - Word2Vec
- Pennington, J. et al. (2014). "GloVe: Global Vectors for Word Representation" - GloVe
- Sparse TF-IDF is well-established in information retrieval and traditional ML

---

**Status**: Final decision - no further changes to embedding selection unless team consensus requires it.
