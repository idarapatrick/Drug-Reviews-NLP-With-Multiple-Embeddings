# Gershom's LSTM Execution Plan

## 🎯 Your Mission
Run all three LSTM notebooks with different embeddings and compare results.

---

## ✅ What's Ready

All three notebooks are created and ready to run:

1. **[lstm_word2vec_skipgram.ipynb](notebooks/lstm_word2vec_skipgram.ipynb)** - Word2Vec Skip-gram (PRIMARY)
2. **[lstm_glove.ipynb](notebooks/lstm_glove.ipynb)** - GloVe (SECONDARY)  
3. **[lstm_tfidf.ipynb](notebooks/lstm_tfidf.ipynb)** - TF-IDF (BASELINE)

---

## 🚀 Execution Steps

### Step 1: Test Your Environment
```powershell
# Check if everything is installed
python -c "import tensorflow, gensim, nltk, sklearn; print('✓ All packages available')"

# Check GPU (optional but helpful)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Step 2: Run First Notebook (Word2Vec)
```powershell
cd notebooks
jupyter notebook lstm_word2vec_skipgram.ipynb
```

**Or in VS Code**: Just open the notebook and run all cells

**Key Settings to Check** (Cell 2):
```python
USE_BIDIRECTIONAL = True   # BiLSTM recommended
CLASSIFICATION_TYPE = 'binary'  # Easier than multiclass
BATCH_SIZE = 32
EPOCHS = 15
```

**Expected Time**: 10-15 min per epoch on CPU, ~2-3 min on GPU

**Expected Output**: `lstm_word2vec_results.json` in notebooks folder

---

### Step 3: Run Second Notebook (GloVe)
```powershell
jupyter notebook lstm_glove.ipynb
```

Same configuration as Word2Vec, just different embedding.

**Expected Output**: `lstm_glove_results.json`

---

### Step 4: Run Third Notebook (TF-IDF)
```powershell
jupyter notebook lstm_tfidf.ipynb
```

**Note**: This one is different! TF-IDF doesn't use an embedding layer.

**Key Settings** (Cell 2):
```python
USE_LSTM = True  # Set False for Dense-only (faster alternative)
USE_BIDIRECTIONAL = True
```

**Expected Output**: `lstm_tfidf_results.json`

---

### Step 5: Compare Results
```powershell
# Go back to project root
cd ..

# Run comparison script
python compare_lstm_results.py
```

This will:
- Load all three result files
- Create comparison charts
- Generate summary report
- Save everything to `notebooks/lstm_embeddings_comparison.png`

---

## 📊 What to Expect

### Typical Performance (Binary Classification)

| Embedding | Expected Accuracy | Training Speed |
|-----------|------------------|----------------|
| Word2Vec  | 75-85%          | Medium         |
| GloVe     | 73-83%          | Medium         |
| TF-IDF    | 70-80%          | Fast           |

BiLSTM typically performs 3-5% better than standard LSTM.

---

## 🔧 Troubleshooting

### "Out of Memory"
```python
# In Cell 2 of any notebook:
BATCH_SIZE = 16  # Reduce from 32
LSTM_UNITS = 64  # Reduce from 128
MAX_SEQUENCE_LENGTH = 150  # Reduce from 200
```

### "Training is too slow"
```python
# Option 1: Use standard LSTM
USE_BIDIRECTIONAL = False

# Option 2: Reduce epochs
EPOCHS = 10  # Instead of 15

# Option 3: For TF-IDF only
USE_LSTM = False  # Use Dense layers only, much faster
```

### "Module not found"
```powershell
# Install missing packages
pip install tensorflow gensim nltk scikit-learn pandas matplotlib seaborn

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 📝 After Running All Experiments

1. Check the generated files:
   - `lstm_word2vec_results.json`
   - `lstm_glove_results.json`
   - `lstm_tfidf_results.json`
   - `lstm_embeddings_comparison.png`
   - `lstm_comparison_summary.txt`

2. Document your findings:
   - Which embedding worked best?
   - Why do you think it performed better?
   - How much better is BiLSTM vs LSTM?
   - Training time differences?

3. Share with team:
   - Compare your LSTM results with Essie's GRU results
   - Discuss which model (GRU vs LSTM) works better
   - Plan final report sections

---

## 💡 Quick Tips

### To Run Faster
- Start with **binary classification** (easier and faster than multiclass)
- Use **smaller batch sizes** if you have limited RAM
- Try **TF-IDF with Dense-only** first (fastest option)

### To Get Better Performance
- Use **BiLSTM** instead of LSTM
- Increase **LSTM_UNITS** to 256
- Try **moderate preprocessing** (already default)
- Fine-tune embeddings: set `trainable=True` in embedding layer

### To Debug
- Check the training plots after each experiment
- If validation loss increases while training loss decreases → overfitting
- If both losses are high → underfitting
- Adjust dropout and model size accordingly

---

## 🎓 Questions to Answer in Your Report

1. **Which embedding is best for LSTM?**
   - Word2Vec / GloVe / TF-IDF?
   - Why?

2. **BiLSTM vs LSTM?**
   - Is the performance gain worth the extra computation?
   - When would you choose one over the other?

3. **How does LSTM compare to other models?**
   - Better/worse than GRU?
   - What are the trade-offs?

4. **Practical insights?**
   - What hyperparameters mattered most?
   - How sensitive is the model to preprocessing?
   - Production recommendations?

---

## 📧 If You Get Stuck

Check:
1. [LSTM_IMPLEMENTATION_GUIDE.md](notebooks/LSTM_IMPLEMENTATION_GUIDE.md) - Detailed guide
2. [lstm_hyperparameters.md](config/lstm_hyperparameters.md) - Configuration options
3. Error messages in notebook output cells

Remember: The notebooks are self-contained. Just run all cells in order!

---

## ✨ Success Criteria

✅ All three notebooks run without errors
✅ Three result JSON files generated  
✅ Comparison script produces charts
✅ You understand which embedding works best and why

---

**Good luck, Gershom! You've got this! 🚀**

*Last updated: February 7, 2026*
