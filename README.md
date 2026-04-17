# 🎬 Sentiment Analysis - IMDb Movie Reviews

NLP project for binary sentiment classification (positive/negative) on IMDb movie reviews.

Built by **Lautaro Rodriguez**.

---

## 📁 Repository Structure

```
nlp-sentiment-imdb/
├── notebooks/
│   └── Analisis_nlp_Lautaro_Rodriguez.ipynb   # Original exploratory notebook
├── src/
│   ├── preprocessing.py      # Text cleaning, NLTK pipeline, spaCy POS Tagging
│   ├── visualization.py      # WordCloud, N-grams, bar charts, confusion matrices
│   └── models/
│       ├── classical.py      # Logistic Regression + SVC with HalvingGridSearchCV
│       ├── rnn_models.py     # RNN, LSTM, GRU with TensorFlow/Keras
│       └── pytorch_model.py  # TextClassifier + EarlyStopping with PyTorch
├── main.py                   # Main pipeline (runs everything)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/tu-usuario/nlp-sentiment-imdb.git
cd nlp-sentiment-imdb

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## ▶️ Usage

### Full pipeline
```bash
python main.py
```

### Using modules individually
```python
from src.preprocessing import apply_cleaning, nltk_pipeline
from src.models.classical import train_logistic_regression
from src.visualization import plot_wordclouds_by_sentiment

# Load and preprocess
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/laurenBack98/Data_set_sentiments_2/refs/heads/main/Review.csv")
df_clean = apply_cleaning(df)
df_nltk  = nltk_pipeline(df_clean)

# Train a model
# ...
```

---

## 📊 Dataset

- **Source:** [Kaggle - IMDb Movie Review NLP](https://www.kaggle.com/datasets/krystalliu152/imbd-movie-reviewnpl)
- **Columns:** `review` (text), `sentiment` (Positive/Negative)
- **Classes:** Balanced (~50% positive, ~50% negative)

---

## 🧠 Implemented Models

| Model                | Framework      | Description                                          |
|----------------------|----------------|------------------------------------------------------|
| Logistic Regression  | scikit-learn   | TF-IDF + LR with HalvingGridSearchCV                 |
| SVC                  | scikit-learn   | TF-IDF + SVM with linear/rbf kernel                  |
| RNN                  | TensorFlow     | Simple recurrent neural network with Embedding layer |
| LSTM                 | TensorFlow     | Long Short-Term Memory network                       |
| GRU                  | TensorFlow     | Gated Recurrent Unit (more efficient)                |
| TextClassifier       | PyTorch        | Dense network + BatchNorm + Dropout + EarlyStopping  |

---

## 📈 Results

Classical models (SVC and Logistic Regression) outperformed the neural networks on this medium-sized dataset, achieving an **F1-Score of ~0.86**.

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| SVC                 | ~0.87    | ~0.86    |
| Logistic Regression | ~0.86    | ~0.86    |
| GRU                 | ~0.76    | ~0.76    |
| LSTM                | ~0.76    | ~0.76    |
| RNN                 | ~0.50    | ~0.50    |

---

## 🔮 Proposed Improvements

- Incorporate pre-trained embeddings (**GloVe** or **BERT**) in the neural network input layer.
- Test on a larger dataset to allow neural networks to surpass classical models.
- Implement fine-tuning of a transformer model (e.g., `bert-base-uncased`).

---

## 🛠️ Tech Stack

- **Python 3.10+**
- pandas, numpy, scikit-learn
- NLTK, spaCy (`en_core_web_sm`)
- TensorFlow / Keras
- PyTorch
- matplotlib, seaborn, wordcloud
