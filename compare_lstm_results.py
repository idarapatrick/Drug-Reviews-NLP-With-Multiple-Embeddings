"""
LSTM Results Comparison Script
Compare performance across Word2Vec, GloVe, and TF-IDF embeddings
Run this after completing all three LSTM notebook experiments
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("LSTM EMBEDDINGS COMPARISON - Gershom")
print("="*70)

# Load results from all three experiments
results_files = {
    'Word2Vec': 'lstm_word2vec_results.json',
    'GloVe': 'lstm_glove_results.json',
    'TF-IDF': 'lstm_tfidf_results.json'
}

all_results = {}
missing_files = []

for embedding_name, filename in results_files.items():
    filepath = os.path.join('notebooks', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_results[embedding_name] = json.load(f)
        print(f"✓ Loaded {embedding_name} results")
    else:
        missing_files.append(embedding_name)
        print(f"✗ Missing {embedding_name} results ({filename})")

if missing_files:
    print(f"\n⚠ Warning: {len(missing_files)} result file(s) not found")
    print("Run the corresponding notebook(s) first:")
    for name in missing_files:
        print(f"  - {name}: {results_files[name].replace('_results.json', '.ipynb')}")
    
    if len(all_results) == 0:
        print("\nNo results to compare. Exiting...")
        exit()

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

# Create comparison DataFrame
comparison_data = []

for embedding_name, results in all_results.items():
    row = {
        'Embedding': embedding_name,
        'Model': results.get('model', 'N/A'),
        'Accuracy': results.get('test_accuracy', results.get('test_mae', 'N/A')),
        'Precision': results.get('precision', 'N/A'),
        'Recall': results.get('recall', 'N/A'),
        'F1-Score': results.get('f1_score', 'N/A'),
        'Epochs': results.get('epochs_trained', 'N/A'),
        'Batch Size': results.get('batch_size', 'N/A'),
    }
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# Display comparison table
print("\n### Performance Metrics ###")
print(df_comparison.to_string(index=False))

# Find best performing embedding
if 'Accuracy' in df_comparison.columns and df_comparison['Accuracy'].dtype in ['float64', 'int64']:
    best_idx = df_comparison['Accuracy'].idxmax()
    best_embedding = df_comparison.loc[best_idx, 'Embedding']
    best_accuracy = df_comparison.loc[best_idx, 'Accuracy']
    
    print(f"\n🏆 Best Performer: {best_embedding}")
    print(f"   Accuracy: {best_accuracy:.4f}")

# Detailed breakdown
print("\n" + "="*70)
print("DETAILED BREAKDOWN")
print("="*70)

for embedding_name, results in all_results.items():
    print(f"\n### {embedding_name} ###")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

# Visualizations
if len(all_results) >= 2:
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy Comparison
    if 'Accuracy' in df_comparison.columns:
        ax = axes[0, 0]
        embeddings = df_comparison['Embedding']
        accuracies = df_comparison['Accuracy']
        
        bars = ax.bar(embeddings, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(embeddings)])
        ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')
    
    # 2. F1-Score Comparison
    if 'F1-Score' in df_comparison.columns:
        ax = axes[0, 1]
        f1_scores = df_comparison['F1-Score']
        
        bars = ax.bar(embeddings, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(embeddings)])
        ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1-Score')
        ax.set_ylim([min(f1_scores) * 0.95, max(f1_scores) * 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')
    
    # 3. Precision-Recall Comparison
    if 'Precision' in df_comparison.columns and 'Recall' in df_comparison.columns:
        ax = axes[1, 0]
        x = range(len(embeddings))
        width = 0.35
        
        precision = df_comparison['Precision']
        recall = df_comparison['Recall']
        
        ax.bar([i - width/2 for i in x], precision, width, label='Precision', color='skyblue')
        ax.bar([i + width/2 for i in x], recall, width, label='Recall', color='lightcoral')
        
        ax.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(embeddings)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Training Efficiency (Epochs)
    if 'Epochs' in df_comparison.columns:
        ax = axes[1, 1]
        epochs = df_comparison['Epochs']
        
        bars = ax.bar(embeddings, epochs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(embeddings)])
        ax.set_title('Training Epochs Until Convergence', fontsize=14, fontweight='bold')
        ax.set_ylabel('Epochs')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('notebooks/lstm_embeddings_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to 'notebooks/lstm_embeddings_comparison.png'")
    plt.show()

# Generate summary report
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

report = f"""
LSTM Models - Embedding Comparison Summary
===========================================

Total Experiments Completed: {len(all_results)}/3

Embeddings Tested:
"""

for embedding_name in all_results.keys():
    report += f"  ✓ {embedding_name}\n"

if missing_files:
    report += "\nPending Experiments:\n"
    for embedding_name in missing_files:
        report += f"  ✗ {embedding_name}\n"

if len(all_results) >= 2:
    report += f"\n\nBest Embedding: {best_embedding}\n"
    report += f"Best Accuracy: {best_accuracy:.4f}\n"

report += f"""

Key Findings:
-------------
1. Review the accuracy comparison chart
2. Check F1-scores for balanced performance
3. Consider precision-recall trade-offs
4. Note training efficiency (epochs to convergence)

Recommendations for Report:
---------------------------
- Document which embedding works best for LSTM
- Explain why certain embeddings perform better
- Compare with other models (GRU, RNN) from team
- Discuss computational trade-offs

Next Steps:
-----------
1. Share results with team for cross-model comparison
2. Write analysis section for final report
3. Create ensemble model if results are close
4. Document hyperparameter tuning insights
"""

print(report)

# Save summary report
with open('notebooks/lstm_comparison_summary.txt', 'w') as f:
    f.write(report)
    f.write("\n\nDetailed Results:\n")
    f.write("="*50 + "\n")
    f.write(df_comparison.to_string(index=False))

print("\n✓ Summary report saved to 'notebooks/lstm_comparison_summary.txt'")

print("\n" + "="*70)
print("COMPARISON COMPLETE!")
print("="*70)
