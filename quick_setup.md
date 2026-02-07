# Quick Setup Guide

Minimal setup instructions for team members implementing GRU/LSTM/Transformer models.

## Prerequisites

1. **Data**: Ensure `/data` exists (or update path in `preprocessing_pipeline.py`)
2. **Dependencies**: Run `pip install -r requirements.txt`

## Step 1: Load Data

Use the shared preprocessing pipeline:

```python
from preprocessing_pipeline import get_data_for_modl

# Load and preprocess data
X_seq, X_tfidf, y, vocab_size, tokenizer, tfidf = get_data_for_model()
```

**Output Interpretation:**
- `X_seq`: (N, 100) - Padded token sequences for GRU/LSTM input
- `X_tfidf`: (N, 2000) - TF-IDF vectors for auxiliary Dense layer input
- `y`: (N,) - Binary sentiment labels (0=Negative, 1=Positive)
- `vocab_size`: Vocabulary size for Embedding layer (`vocab_size` parameter)
- `tokenizer`: Fitted Keras Tokenizer for inference
- `tfidf`: Fitted TfidfVectorizer for inference

## Step 2: Build Your Model

### Hybrid Architecture Template

```python
from tensorflow.keras.layers import Input, Embedding, Dense, GRU, Concatenate
from tensorflow.keras.models import Model

# Define inputs
seq_input = Input(shape=(100,), name='sequence_input')
tfidf_input = Input(shape=(2000,), name='tfidf_input')

# Sequence branch
embed = Embedding(vocab_size, 100)(seq_input)
gru_out = GRU(64, return_sequences=False)(embed)

# TF-IDF branch
tfidf_dense = Dense(32, activation='relu')(tfidf_input)

# Merge branches
merged = Concatenate()([gru_out, tfidf_dense])

# Output
output = Dense(1, activation='sigmoid')(merged)

# Create model
model = Model(inputs=[seq_input, tfidf_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Step 3: Train Your Model

```python
# Train on both inputs
history = model.fit(
    [X_seq, X_tfidf], 
    y, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2
)
```

## Step 4: Evaluate

```python
# Load test data (if using another dataset)
from preprocessing_pipeline import load_and_clean_csv
X_test_seq, X_test_tfidf, y_test, _, _, _ = get_data_for_model('/path/to/test/data.csv')

# Evaluate
loss, accuracy = model.evaluate([X_test_seq, X_test_tfidf], y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Configuration Reference

### In `preprocessing_pipeline.py`, these constants define data processing:
- `MAX_LEN = 100`: Sequence length (pad or truncate to this)
- `MAX_WORDS = 20000`: Tokenizer vocabulary size
- `TFIDF_FEATURES = 2000`: TF-IDF dimensions

### In Your Notebook, suggested hyperparameters:
- **Embedding dimension**: 100
- **GRU units**: 64
- **Dense units** (TF-IDF branch): 32
- **Dropout**: 0.3 (optional)
- **Epochs**: 20+
- **Batch size**: 32
- **Optimizer**: Adam (default)
- **Loss**: binary_crossentropy
- **Metrics**: accuracy

## Troubleshooting

### FileNotFoundError: `/data/drugsComTrain_raw.csv`
Update the path in the function call:
```python
X_seq, X_tfidf, y, vocab_size, tokenizer, tfidf = get_data_for_model(
    train_path='/your/actual/path/to/file.csv'
)
```

### Memory Issues
Reduce batch size or use a smaller subset of data:
```python
X_seq_sample = X_seq[:50000]
X_tfidf_sample = X_tfidf[:50000]
y_sample = y[:50000]
```

### Poor Model Performance
- Increase epochs (try 50+)
- Adjust GRU units (try 128)
- Add dropout layers
- Ensure both inputs are being used
- Check target class balance: `np.bincount(y)`

## Additional Embeddings (Optional)

To use Word2Vec or GloVe embeddings in place of the default Embedding layer:

```python
from embeddings.word2vec_embedding import Word2VecEmbedding
from embeddings.glove_embedding import GloVeEmbedding

# Train Word2Vec embedding
w2v = Word2VecEmbedding(embedding_dim=100)
w2v.train(tokenizer.word_index)  # Or on actual texts

# Get embedding matrix for Embedding layer
embedding_matrix = w2v.get_embedding_matrix(tokenizer, vocab_size)

# Use in model
embed = Embedding(
    vocab_size, 
    100, 
    weights=[embedding_matrix], 
    trainable=False
)(seq_input)
```

See `embeddings/` folder for detailed embedding documentation.

---

**Quick Summary:**
1. Import `get_data_for_model()`
2. Build Hybrid model with two input branches
3. Train with `[X_seq, X_tfidf]` inputs
4. Evaluate and document results
