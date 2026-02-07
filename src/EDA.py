"""
EDA Standard - Data Exploration for Drug Review Dataset

Validates the new dataset schema and provides standard visualizations.
Run this to ensure the data is properly formatted and ready for modeling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing_pipeline import load_and_clean_csv

DATA_PATH = '/data/drugsComTrain_raw.csv'


def run_eda():
    """Run exploratory data analysis on the loaded data."""
    try:
        print("Loading data...")
        df = load_and_clean_csv(DATA_PATH)
        print(f"Data Loaded. Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        # 1. Target Distribution
        print("\n1. Sentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        print(f"{sentiment_counts}")
        print(f"\nPositive (Rating >= 7): {sentiment_counts.get(1, 0)} samples")
        print(f"Negative (Rating <= 4): {sentiment_counts.get(0, 0)} samples")
        
        # Plot sentiment distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sentiment_counts.plot(kind='bar', ax=ax, color=['coral', 'skyblue'])
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Distribution')
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'], rotation=0)
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=100, bbox_inches='tight')
        print("\nSaved: sentiment_distribution.png")
        plt.show()
        
        # 2. Sequence Length Check
        print("\n2. Review Length Statistics:")
        df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))
        length_stats = df['word_count'].describe()
        print(length_stats)
        
        # Plot review length distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['word_count'], bins=50, range=(0, 500), color='steelblue', edgecolor='black')
        ax.axvline(100, color='red', linestyle='--', linewidth=2, label='Cutoff (MAX_LEN=100)')
        ax.set_xlabel('Review Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('Review Length Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig('review_length_distribution.png', dpi=100, bbox_inches='tight')
        print("\nSaved: review_length_distribution.png")
        plt.show()
        
        # 3. Additional statistics
        print("\n3. Additional Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Missing values in 'review': {df['review'].isna().sum()}")
        print(f"Missing values in 'rating': {df['rating'].isna().sum()}")
        
        # Rating distribution before filtering
        print("\n4. Rating Distribution (Full, Before Binary Conversion):")
        rating_dist = df['rating'].value_counts().sort_index()
        print(rating_dist)
        
        # Words per sentiment
        print("\n5. Average Review Length by Sentiment:")
        sentiment_length = df.groupby('sentiment')['word_count'].agg(['mean', 'median', 'std', 'min', 'max'])
        print(sentiment_length)
        
        print("\n✓ EDA completed successfully!")
        
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Please ensure the file exists or update DATA_PATH.")
    except Exception as e:
        print(f"ERROR: EDA Failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_eda()
