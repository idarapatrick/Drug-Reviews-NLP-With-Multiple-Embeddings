# LSTM/BiLSTM Hyperparameter Recommendations

## Recommended Configurations for LSTM Models

### Configuration 1: Small/Fast (Quick Experiments)
```python
# Model Architecture
USE_BIDIRECTIONAL = False
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT = 0.1

# Training
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3

# Sequences
MAX_SEQUENCE_LENGTH = 150
VOCAB_SIZE = 5000

# Estimated training time: 5-10 minutes per epoch on CPU
```

### Configuration 2: Medium/Balanced (Recommended)
```python
# Model Architecture
USE_BIDIRECTIONAL = True  # BiLSTM for better performance
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2

# Training
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3

# Sequences
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000

# Estimated training time: 10-15 minutes per epoch on CPU, 2-3 minutes on GPU
```

### Configuration 3: Large/Production (Best Performance)
```python
# Model Architecture
USE_BIDIRECTIONAL = True
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005  # Lower LR for stability
PATIENCE = 5

# Sequences
MAX_SEQUENCE_LENGTH = 300
VOCAB_SIZE = 15000

# Estimated training time: 15-25 minutes per epoch on CPU, 3-5 minutes on GPU
# Requires: ~8GB RAM, ideally GPU
```

---

## Embedding-Specific Recommendations

### For Word2Vec Embeddings
```python
# Word2Vec Config
EMBEDDING_CONFIG = 'skipgram_medium'  # 200-dim
EMBEDDING_DIM = 200

# Model adjustments
LSTM_UNITS = 128  # Good match for 200-dim embeddings
MAX_SEQUENCE_LENGTH = 200
TRAINABLE_EMBEDDINGS = False  # Keep pre-trained weights frozen

# Works well with moderate preprocessing
PREPROCESSING_CONFIG = 'moderate'
```

### For GloVe Embeddings
```python
# GloVe Config
EMBEDDING_CONFIG = 'medium'  # 200-dim
EMBEDDING_DIM = 200

# Model adjustments
LSTM_UNITS = 128
MAX_SEQUENCE_LENGTH = 200
TRAINABLE_EMBEDDINGS = False

# Works well with moderate preprocessing
PREPROCESSING_CONFIG = 'moderate'
```

### For TF-IDF Embeddings
```python
# TF-IDF Config
EMBEDDING_CONFIG = 'balanced'  # 5000 features
TFIDF_MAX_FEATURES = 5000

# Architecture (NO embedding layer needed!)
# Input shape: (batch_size, 5000)
# Goes directly to Dense layers

DENSE_LAYER_1 = 256
DENSE_LAYER_2 = 128
DROPOUT_RATE = 0.4  # Higher dropout for TF-IDF

# Can use simpler model or even skip LSTM
# Option: Use Dense layers only for faster training
```

---

## Classification Task Recommendations

### Binary Classification (Positive/Negative Reviews)
```python
CLASSIFICATION_TYPE = 'binary'
THRESHOLD = 6  # ratings >= 6 = positive (1), < 6 = negative (0)

# Output layer
OUTPUT_UNITS = 1
OUTPUT_ACTIVATION = 'sigmoid'
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']

# Expected accuracy: 75-85% (BiLSTM + Word2Vec)
```

### Multi-class Classification (Rating 1-10)
```python
CLASSIFICATION_TYPE = 'multiclass'
NUM_CLASSES = 10

# Output layer
OUTPUT_UNITS = 10
OUTPUT_ACTIVATION = 'softmax'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

# Expected accuracy: 45-55% (harder task)
# May need larger model and more training
```

### Regression (Predict Exact Rating)
```python
CLASSIFICATION_TYPE = 'regression'

# Output layer
OUTPUT_UNITS = 1
OUTPUT_ACTIVATION = 'linear'
LOSS = 'mse'
METRICS = ['mae']

# Expected MAE: 1.0-2.0
# R²: 0.5-0.7
```

---

## Layer Architecture Options

### Option 1: Two-Layer BiLSTM (Recommended)
```python
model.add(Embedding(...))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Total params: ~500K-800K
# Training time: Medium
# Performance: Best
```

### Option 2: Single-Layer BiLSTM (Faster)
```python
model.add(Embedding(...))
model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Total params: ~300K-500K
# Training time: Fast
# Performance: Good (2-3% lower than two-layer)
```

### Option 3: Three-Layer LSTM (Deep, for large datasets)
```python
model.add(Embedding(...))
model.add(LSTM(256, return_sequences=True, dropout=0.3))
model.add(LSTM(128, return_sequences=True, dropout=0.3))
model.add(LSTM(64, dropout=0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Total params: ~1M+
# Training time: Slow
# Performance: May overfit on small datasets
```

### Option 4: LSTM with Attention (Advanced)
```python
# Requires custom attention layer
from tensorflow.keras.layers import Attention

model.add(Embedding(...))
lstm_out = model.add(LSTM(128, return_sequences=True))
attention = Attention()([lstm_out, lstm_out])
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Better performance but more complex
```

---

## Training Callbacks Configuration

### Recommended Callbacks
```python
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)

callbacks = [
    # Stop training when validation loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        filepath='best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Optional: TensorBoard logging
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]
```

---

## Optimizer Options

### Option 1: Adam (Recommended, Default)
```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)

# Fast convergence, works well for most cases
```

### Option 2: SGD with Momentum (Alternative)
```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

# May need learning rate scheduling
# Can achieve better final performance but slower
```

### Option 3: RMSprop (For RNNs)
```python
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9
)

# Good for RNNs/LSTMs, alternative to Adam
```

---

## Performance Tuning Guide

### If Model is Underfitting (Low Training Accuracy)
```python
# Increase model capacity
LSTM_UNITS = 256  # from 128
DENSE_UNITS = 128  # from 64

# Add more layers
# Add a third LSTM layer

# Increase sequence length
MAX_SEQUENCE_LENGTH = 300  # from 200

# Increase vocabulary
VOCAB_SIZE = 15000  # from 10000

# Train longer
EPOCHS = 25  # from 15

# Reduce regularization
DROPOUT_RATE = 0.2  # from 0.3
```

### If Model is Overfitting (Train Acc >> Val Acc)
```python
# Increase regularization
DROPOUT_RATE = 0.4  # from 0.3
RECURRENT_DROPOUT = 0.3  # from 0.2

# Add more dropout layers
# After each Dense layer

# Reduce model capacity
LSTM_UNITS = 64  # from 128

# Use early stopping
PATIENCE = 2  # from 3

# Reduce vocabulary
VOCAB_SIZE = 5000  # from 10000

# Data augmentation (if possible)
```

### If Training is Too Slow
```python
# Use standard LSTM instead of BiLSTM
USE_BIDIRECTIONAL = False

# Reduce batch size (paradoxically can be faster)
BATCH_SIZE = 64  # from 32

# Reduce sequence length
MAX_SEQUENCE_LENGTH = 150  # from 200

# Reduce vocabulary
VOCAB_SIZE = 5000  # from 10000

# Use GPU if available
# Enable mixed precision training
```

---

## Memory Usage Estimation

### Small Model (~500MB RAM)
```python
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 150
LSTM_UNITS = 64
BATCH_SIZE = 16
```

### Medium Model (~2GB RAM)
```python
VOCAB_SIZE = 10000
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 200
LSTM_UNITS = 128
BATCH_SIZE = 32
```

### Large Model (~8GB RAM)
```python
VOCAB_SIZE = 15000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 300
LSTM_UNITS = 256
BATCH_SIZE = 64
```

---

## Experiment Tracking Template

```python
experiment_log = {
    'experiment_id': 'lstm_001',
    'date': '2026-02-07',
    'researcher': 'Gershom',
    
    # Model config
    'model_type': 'BiLSTM',
    'embedding': 'word2vec_skipgram_medium',
    'lstm_units': 128,
    'layers': 2,
    'dropout': 0.3,
    
    # Training config
    'batch_size': 32,
    'epochs_run': 12,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    
    # Data config
    'preprocessing': 'moderate',
    'vocab_size': 10000,
    'max_seq_length': 200,
    'train_samples': 18000,
    'val_samples': 4500,
    'test_samples': 4500,
    
    # Results
    'test_accuracy': 0.XX,
    'test_precision': 0.XX,
    'test_recall': 0.XX,
    'test_f1': 0.XX,
    'training_time_minutes': XX,
    
    # Notes
    'notes': 'Early stopping at epoch 12. Good convergence.',
    'issues': 'None',
    'next_steps': 'Try GloVe embedding'
}
```

---

**Reference**: Use this guide while running your experiments in the notebooks.

*Last updated: February 7, 2026*
