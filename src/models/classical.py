"""
models/classical.py
Classical ML models: Logistic Regression and SVC.
Each function returns the best trained estimator.
"""

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
#  LOGISTIC REGRESSION
# =============================================================================

def train_logistic_regression(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 0,
) -> tuple:
    """
    Trains a TF-IDF + Logistic Regression pipeline with HalvingGridSearchCV.

    Returns
    -------
    model       : best trained estimator
    x_test      : test features
    y_test      : test labels
    y_pred      : predictions on test set
    """
    x = features[text_col]
    y = features[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression()),
    ])

    hyperparams = {
        'vectorizer__max_features': [1000, 2000, 5000, 10000],
        'vectorizer__ngram_range':  [(1, 1), (1, 2)],
        'vectorizer__sublinear_tf': [True],
        'classifier__class_weight': ['balanced', None],
        'classifier__penalty':      ['l1', 'l2'],
        'classifier__solver':       ['liblinear'],
        'classifier__C':            [0.01, 0.1, 1, 10, 100],
    }

    search = HalvingGridSearchCV(
        pipeline, param_grid=hyperparams,
        cv=5, n_jobs=-1, verbose=verbose
    )
    search.fit(x_train, y_train)

    model  = search.best_estimator_
    y_pred = model.predict(x_test)

    print(f"[Logistic Regression] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Best parameters: {search.best_params_}")

    return model, x_test, y_test, y_pred


# =============================================================================
#  SVC (Support Vector Machine)
# =============================================================================

def train_svc(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 0,
) -> tuple:
    """
    Trains a TF-IDF + SVC pipeline with HalvingGridSearchCV.

    Returns
    -------
    model       : best trained estimator
    x_test      : test features
    y_test      : test labels
    y_pred      : predictions on test set
    """
    x = features[text_col]
    y = features[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('model_svm',  SVC()),
    ])

    hyperparams = {
        'vectorizer__max_features': [1000, 3000, 5000],
        'vectorizer__ngram_range':  [(1, 1), (1, 2)],
        'vectorizer__sublinear_tf': [True, False],
        'model_svm__C':             [0.1, 1, 10],
        'model_svm__kernel':        ['linear', 'rbf'],
    }

    search = HalvingGridSearchCV(
        pipeline, param_grid=hyperparams,
        scoring='f1', cv=5, n_jobs=-1, verbose=verbose
    )
    search.fit(x_train, y_train)

    model  = search.best_estimator_
    model.fit(x_train, y_train)   # retrain the best estimator on all training data
    y_pred = model.predict(x_test)

    print(f"[SVC] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Best parameters: {search.best_params_}")

    return model, x_test, y_test, y_pred
