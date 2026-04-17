"""
preprocessing.py
Text cleaning, tokenization, stopword removal, lemmatization (NLTK) and POS Tagging (spaCy).
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# ── Download NLTK resources (first time only) ─────────────────────────────────
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)


# =============================================================================
#  REGEX-BASED CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Removes non-alphabetic characters, parentheses, and extra whitespace.
    Returns the text in lowercase.
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'[()]',      ' ', text)
    text = re.sub(r'\s+',       ' ', text)
    return text.lower().strip()


def apply_cleaning(df: pd.DataFrame, col: str = 'review') -> pd.DataFrame:
    """
    Applies clean_text to column `col` and adds 'review_clean'.
    Also maps the 'sentiment' column to binary values in 'sentiment_map'.
    """
    df = df.copy()
    df['review_clean']   = df[col].apply(clean_text)
    df['sentiment_map']  = df['sentiment'].map({'Positive': 1, 'Negative': 0})
    return df


# =============================================================================
#  NLTK PIPELINE: tokenization → stopword removal → lemmatization
# =============================================================================

_stop_words  = set(stopwords.words('english'))
_lemmatizer  = WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    """Tokenizes a string of text."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Removes stopwords from a list of tokens."""
    return [w for w in tokens if w not in _stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatizes a list of tokens."""
    return [_lemmatizer.lemmatize(w) for w in tokens]


def nltk_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full NLTK pipeline on 'review_clean'.
    Returns a DataFrame with 'review_lematizer' (list of tokens)
    and 'review_lematizer_str' (space-joined string).
    """
    df = df.copy()
    df['review_tokenized']  = df['review_clean'].apply(tokenize)
    df['review_stopwords']  = df['review_tokenized'].apply(remove_stopwords)
    df['review_lematizer']  = df['review_stopwords'].apply(lemmatize)
    df['review_lematizer_str'] = df['review_lematizer'].apply(lambda x: ' '.join(x))

    # Keep only the relevant columns
    return df.drop(columns=['review_tokenized', 'review_stopwords'])


# =============================================================================
#  spaCy PIPELINE: contextual lemmatization + POS Tagging
# =============================================================================

def spacy_lemmatize(df: pd.DataFrame,
                    col: str = 'review_clean',
                    batch_size: int = 500) -> list[str]:
    """
    Lemmatizes using spaCy (without NER or parser for faster processing).
    Returns a list of lemmatized strings.
    """
    nlp = spacy.load('en_core_web_sm')
    result = []
    for doc in nlp.pipe(df[col], disable=['ner', 'parser'], batch_size=batch_size):
        result.append(' '.join([
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
        ]))
    return result


def pos_tagging(texts: list[str], batch_size: int = 500) -> pd.DataFrame:
    """
    Applies POS Tagging with spaCy and filters VERB, ADJ, and NOUN (highest semantic value).
    Returns a DataFrame with columns: review_id, token, pos_tag.
    """
    nlp = spacy.load('en_core_web_sm')
    rows = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                rows.append({
                    'review_id': i,
                    'token':     token.text,
                    'pos_tag':   token.pos_
                })
    df = pd.DataFrame(rows)
    # Keep only categories with highest semantic value
    return df[df['pos_tag'].isin(['VERB', 'ADJ', 'NOUN'])]


def build_features(concat_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups tokens by review_id and merges with the original DataFrame
    to recover 'sentiment_map'. Handles a possible 1-record mismatch
    using merge() + notna().
    """
    features = pos_df.groupby('review_id')['token'].apply(' '.join).reset_index()

    concat_df = concat_df.reset_index(drop=True)
    concat_df.index.name = 'review_id'
    concat_df = concat_df.reset_index()

    result = concat_df.merge(features, on='review_id', how='left')
    result = result[result['token'].notna()].reset_index(drop=True)

    return result[['review_id', 'token', 'sentiment_map']].copy()
