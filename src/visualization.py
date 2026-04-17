"""
visualization.py
All project visualizations: WordCloud, N-grams,
word frequency charts, confusion matrices, and training curves.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix


# =============================================================================
#  WORDCLOUDS
# =============================================================================

def plot_wordcloud(text: str, title: str = 'WordCloud',
                   width: int = 800, height: int = 400,
                   max_words: int = 200) -> None:
    """Generates and displays a word cloud from a string."""
    wc = WordCloud(
        width=width, height=height,
        background_color='white',
        max_words=max_words,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_wordclouds_by_sentiment(df: pd.DataFrame,
                                  text_col: str = 'review_lematizer_str',
                                  sentiment_col: str = 'sentiment_map') -> None:
    """
    Displays three WordClouds: general, positive, and negative.
    """
    text_all = ' '.join(df[text_col])
    text_pos = ' '.join(df[df[sentiment_col] == 1][text_col])
    text_neg = ' '.join(df[df[sentiment_col] == 0][text_col])

    plot_wordcloud(text_all, title='WordCloud - General')
    plot_wordcloud(text_pos, title='WordCloud - Positive Reviews')
    plot_wordcloud(text_neg, title='WordCloud - Negative Reviews')


# =============================================================================
#  N-GRAMS
# =============================================================================

def plot_ngram_wordcloud(df: pd.DataFrame,
                          text_col: str = 'review_lematizer_str',
                          sentiment_col: str = 'sentiment_map',
                          sentiment_value: int | None = None,
                          ngram_range: tuple = (3, 3),
                          stop_words: list | None = None,
                          title: str = 'N-Grams WordCloud') -> None:
    """
    Generates a WordCloud of N-grams.
    sentiment_value: 1=positive, 0=negative, None=general.
    """
    if sentiment_value is not None:
        subset = df[df[sentiment_col] == sentiment_value]
    else:
        subset = df

    vec = CountVectorizer(ngram_range=ngram_range,
                          stop_words=stop_words or 'english')
    X   = vec.fit_transform(subset[text_col])

    freq = dict(zip(vec.get_feature_names_out(), X.toarray().sum(axis=0)))

    wc = WordCloud(width=800, height=400, background_color='white') \
         .generate_from_frequencies(freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
#  WORD FREQUENCY (BAR CHARTS)
# =============================================================================

def plot_top_words(df: pd.DataFrame,
                   token_col: str = 'review_lematizer',
                   sentiment_col: str = 'sentiment_map',
                   top_n: int = 10) -> None:
    """
    Displays three bar charts: general, positive, and negative.
    token_col must contain lists of tokens.
    """
    configs = [
        (None,  'skyblue', 'Top words - General'),
        (1,     'green',   'Top words - Positive'),
        (0,     'red',     'Top words - Negative'),
    ]

    for val, color, title in configs:
        subset = df if val is None else df[df[sentiment_col] == val]
        tokens = [t for lst in subset[token_col] for t in lst]
        words, freqs = zip(*Counter(tokens).most_common(top_n))

        plt.figure(figsize=(10, 6))
        plt.bar(words, freqs, color=color)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('Lemmatized words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# =============================================================================
#  TRAINING CURVES (RNN / LSTM / GRU)
# =============================================================================

def plot_history(history, model_name: str = 'Model') -> None:
    """
    Plots accuracy and loss curves from a Keras training history object.
    """
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(history.history['accuracy'],     label='Train',      linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train',      linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
    axes[1].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.suptitle(f'{model_name} Performance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(models_results: list[tuple]) -> None:
    """
    Displays confusion matrices for multiple models in a single figure.

    Parameters
    ----------
    models_results : list of tuples (name, y_true, y_pred)
    """
    n = len(models_results)
    sns.set_theme(style='white')
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, y_true, y_pred) in zip(axes, models_results):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    ax=ax)
        ax.set_title(f'Confusion Matrix\n{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Model Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  METRICS SUMMARY TABLE
# =============================================================================

def metrics_table(models_results: list[tuple]) -> pd.DataFrame:
    """
    Builds a comparative metrics table.

    Parameters
    ----------
    models_results : list of tuples (name, y_true, y_pred)

    Returns
    -------
    DataFrame sorted by F1-Score descending.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    rows = []
    for name, y_true, y_pred in models_results:
        rows.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_true, y_pred),  4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall':    round(recall_score(y_true, y_pred),    4),
            'F1-Score':  round(f1_score(y_true, y_pred),        4),
        })

    return pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
