# Hyperparameter Recommendations for Different Models

## GRU Model Hyperparameters

### For Word2Vec/GloVe Embeddings
```python
# Small model (quick experiments)
batch_size = 16
epochs = 5
gru_units = 64
dropout = 0.2
learning_rate = 0.001

# Medium model (recommended)
batch_size = 32
epochs = 10
gru_units = 128
dropout = 0.3
learning_rate = 0.001

# Large model (if you have compute resources)
batch_size = 64
epochs = 15
gru_units = 256
dropout = 0.3
learning_rate = 0.0005
```

### For TF-IDF Embeddings
```python
# Dense model (TF-IDF is already sparse)
batch_size = 32
epochs = 20
layers = 2-3 dense layers
units = [128, 64]
dropout = 0.4
learning_rate = 0.001
```

## Embedding Configuration Options

### TF-IDF
- `conservative`: max_features=3000, unigrams only
- `balanced`: max_features=5000, unigrams+bigrams (recommended)
- `aggressive`: max_features=10000, unigrams+bigrams+trigrams

### Word2Vec
- `skipgram_small`: 100-dim, 5 epochs (experiments)
- `skipgram_medium`: 200-dim, 10 epochs (recommended)
- `skipgram_large`: 300-dim, 15 epochs (production)
- `cbow_small/medium/large`: Same sizes, CBOW method

### GloVe
- `small`: 100-dim, 10 iterations
- `medium`: 200-dim, 15 iterations (recommended)
- `large`: 300-dim, 20 iterations

## Preprocessing Configurations

### minimal
- Lowercase
- Remove URLs
- Remove HTML

**Best for**: Word2Vec and GloVe (they preserve word context)

### moderate (recommended)
- All of minimal, plus:
- Remove punctuation
- Remove stopwords

**Best for**: Most cases, balanced approach

### aggressive
- All of moderate, plus:
- Remove numbers
- Lemmatization
- Minimum token length = 2

**Best for**: TF-IDF and traditional ML models, when vocabulary size is important

## Model Architecture Templates

### Simple GRU (for quick testing)
```
Input (embedding_dim,)
  ↓
GRU(64 units)
  ↓
Dropout(0.2)
  ↓
Dense(32)
  ↓
Dense(1, sigmoid)
```

### Medium GRU (recommended)
```
Input (embedding_dim,)
  ↓
GRU(128 units, return_sequences=True)
  ↓
Dropout(0.3)
  ↓
GRU(64 units)
  ↓
Dropout(0.3)
  ↓
Dense(64)
  ↓
Dense(1, sigmoid)
```

### Advanced GRU (bidirectional)
```
Input (embedding_dim,)
  ↓
Bidirectional GRU(128 units, return_sequences=True)
  ↓
Dropout(0.3)
  ↓
GRU(64 units)
  ↓
Dropout(0.3)
  ↓
Dense(64)
  ↓
Dense(1, sigmoid)
```

## Training Tips

1. **Batch Size**: Start with 32, adjust based on memory
2. **Learning Rate**: 0.001 is safe default, try 0.0005 if overf itting
3. **Epochs**: Monitor validation loss, stop if it increases
4. **Early Stopping**: Consider adding patience=3 to avoid overfitting
5. **Class Imbalance**: Use `class_weight` parameter if classes are imbalanced

## Recommended Experiment Matrix for GRU

| Embedding | Config | Preprocessing | GRU Units | Expected Accuracy |
|-----------|--------|--------------|-----------|-------------------|
| Word2Vec | skipgram_medium | moderate | 128 | 75-85% |
| Word2Vec | skipgram_medium | minimal | 128 | 72-82% |
| Word2Vec | cbow_medium | moderate | 128 | 73-83% |
| GloVe | medium | moderate | 128 | 75-85% |
| TF-IDF | balanced | aggressive | 128 | 70-80% |

(Actual numbers depend on your specific dataset!)

---

**Note**: These are starting points. Always experiment and record your results!
