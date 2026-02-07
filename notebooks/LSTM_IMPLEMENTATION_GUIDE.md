# LSTM Implementation Guide - Gershom

## Overview
This guide helps you implement LSTM and BiLSTM models with the three selected embeddings for drug review classification.

---

## Your Notebooks (To Create)

### ✅ 1. `lstm_word2vec_skipgram.ipynb` - CREATED
**Status**: Ready to run!  
**Embedding**: Word2Vec Skip-gram (200-dim)  
**Purpose**: Primary semantic embedding approach

### 🔄 2. `lstm_glove.ipynb` - TO CREATE
**Embedding**: GloVe (200-dim)  
**Purpose**: Global context patterns  
**Notes**: Similar structure to Word2Vec notebook, just swap embedding module

### 🔄 3. `lstm_tfidf.ipynb` - TO CREATE  
**Embedding**: TF-IDF (5000 features)  
**Purpose**: Statistical baseline  
**Notes**: Different preprocessing - TF-IDF uses raw text, not tokens

---

## Quick Start

### Step 1: Run the First Notebook
```bash
# Open Jupyter or VS Code
cd notebooks
jupyter notebook lstm_word2vec_skipgram.ipynb
```

### Step 2: Key Configuration Options

**Choose Classification Type:**
```python
CLASSIFICATION_TYPE = 'binary'      # Positive vs Negative reviews
CLASSIFICATION_TYPE = 'multiclass'  # Rating classes 1-10
CLASSIFICATION_TYPE = 'regression'  # Predict exact rating
```

**LSTM vs BiLSTM:**
```python
USE_BIDIRECTIONAL = True   # BiLSTM (recommended, better performance)
USE_BIDIRECTIONAL = False  # Standard LSTM (faster)
```

**Model Size:**
```python
# Small (quick experiments)
LSTM_UNITS = 64
DENSE_UNITS = 32
BATCH_SIZE = 16

# Medium (recommended)
LSTM_UNITS = 128
DENSE_UNITS = 64
BATCH_SIZE = 32

# Large (if you have GPU)
LSTM_UNITS = 256
DENSE_UNITS = 128
BATCH_SIZE = 64
```

---

## LSTM Architecture Explained

### Standard LSTM (2 layers)
```
Input Sequences (200 words)
    ↓
Embedding Layer (Word2Vec 200-dim, frozen)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (1 unit, sigmoid for binary)
```

### BiLSTM (Bidirectional)
```
Input Sequences (200 words)
    ↓
Embedding Layer (Word2Vec 200-dim, frozen)
    ↓
BiLSTM Layer 1 (128 units × 2 directions = 256 output)
    ↓
BiLSTM Layer 2 (64 units × 2 directions = 128 output)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (1 unit, sigmoid)
```

**Why BiLSTM?**
- Reads text forwards AND backwards
- Better contextual understanding
- Often 2-5% better accuracy than standard LSTM
- Slightly slower training (worth it!)

---

## Embedding Differences

### Word2Vec (Current Notebook)
```python
# Requires tokenized input
train_tokens = preprocessor.get_tokens_batch(train_texts)
w2v_model.fit(train_tokens)

# Token-based embeddings
embedding_matrix from Word2Vec vectors
```

### GloVe (Next Notebook)
```python
# Also requires tokenized input
train_tokens = preprocessor.get_tokens_batch(train_texts)
glove_model.fit(train_tokens)

# Similar to Word2Vec in structure
```

### TF-IDF (Third Notebook)
```python
# Uses RAW text (no tokenization)
tfidf_model.fit(train_texts)  # Direct strings
X_train = tfidf_model.transform_dense(train_texts)

# NO embedding layer needed!
# Feed directly to Dense layers
```

---

## Expected Performance Ranges

### Binary Classification (Positive/Negative)

| Model | Embedding | Expected Accuracy |
|-------|-----------|-------------------|
| BiLSTM | Word2Vec | 75-85% |
| BiLSTM | GloVe | 73-83% |
| BiLSTM | TF-IDF | 70-80% |
| LSTM | Word2Vec | 72-82% |
| LSTM | GloVe | 70-80% |
| LSTM | TF-IDF | 68-78% |

### Multi-class (Rating 1-10)

| Model | Embedding | Expected Accuracy |
|-------|-----------|-------------------|
| BiLSTM | Word2Vec | 45-55% |
| BiLSTM | GloVe | 43-53% |
| BiLSTM | TF-IDF | 40-50% |

*Note: Multi-class is harder due to fine-grained rating distinctions*

---

## Training Tips

### 1. Monitor Overfitting
```python
# If validation loss increases while training loss decreases:
- Increase dropout_rate (0.3 → 0.4 → 0.5)
- Add more dropout layers
- Use early stopping (already configured)
- Reduce model size
```

### 2. Improve Performance
```python
# If accuracy is too low:
- Increase LSTM_UNITS (128 → 256)
- Add more LSTM layers
- Increase MAX_SEQUENCE_LENGTH (200 → 300)
- Try different preprocessing configs
- Fine-tune embeddings (set trainable=True)
```

### 3. Speed Up Training
```python
# If training is too slow:
- Decrease BATCH_SIZE (32 → 64)
- Use standard LSTM instead of BiLSTM
- Reduce MAX_SEQUENCE_LENGTH
- Use GPU if available
```

---

## Results Tracking

### After Each Experiment, Record:

```python
results = {
    'model': 'BiLSTM',
    'embedding': 'word2vec_skipgram_medium',
    'test_accuracy': 0.XX,
    'precision': 0.XX,
    'recall': 0.XX,
    'f1_score': 0.XX,
    'training_time': 'XX minutes',
    'notes': 'Any observations'
}
```

### Save to JSON
```python
import json
with open('lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Error: "Out of Memory"
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8

# Reduce vocabulary
VOCAB_SIZE = 5000  # instead of 10000

# Reduce sequence length
MAX_SEQUENCE_LENGTH = 100  # instead of 200
```

### Error: "Word not in vocabulary"
```python
# Already handled in the notebook with:
word_to_idx.get(token, word_to_idx['<UNK>'])
```

### Warning: "Model not converging"
```python
# Increase learning rate
LEARNING_RATE = 0.01  # instead of 0.001

# Or decrease if loss is unstable
LEARNING_RATE = 0.0001
```

---

## Comparison Matrix Template

After completing all 3 notebooks, create this comparison:

| Metric | Word2Vec | GloVe | TF-IDF |
|--------|----------|-------|--------|
| **Accuracy** | X.XX | X.XX | X.XX |
| **Precision** | X.XX | X.XX | X.XX |
| **Recall** | X.XX | X.XX | X.XX |
| **F1-Score** | X.XX | X.XX | X.XX |
| **Training Time** | Xm | Xm | Xm |
| **Parameters** | XXX,XXX | XXX,XXX | XXX,XXX |

**Best Performer**: [embedding_name]  
**Why**: [brief explanation]

---

## Next Steps

### Immediate (Today):
1. ✅ Run `lstm_word2vec_skipgram.ipynb`
2. Fix any errors and tune hyperparameters
3. Record results

### This Week:
1. Create `lstm_glove.ipynb` (copy Word2Vec notebook, swap embedding)
2. Create `lstm_tfidf.ipynb` (different architecture, no embedding layer)
3. Run all experiments

### Final Phase:
1. Create comparison visualizations
2. Write analysis section for report
3. Share results with team (Essie for GRU comparison)

---

## Helpful Commands

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install missing packages
pip install tensorflow gensim nltk scikit-learn

# Open notebook
jupyter notebook lstm_word2vec_skipgram.ipynb
```

---

## Questions to Answer in Your Report

1. **Which embedding works best with LSTM?** Word2Vec/GloVe/TF-IDF?
2. **BiLSTM vs LSTM**: Is the performance gain worth the extra computation?
3. **How does LSTM compare to other models** (GRU, RNN) on this task?
4. **What's the impact of preprocessing?** Moderate vs aggressive?
5. **Optimal hyperparameters**: What configuration works best?

---

**Good luck with your LSTM implementation, Gershom!** 🚀

*Last updated: February 7, 2026*
