# Fake News Detector

A machine learning system for classifying news articles as fake or real using TF-IDF vectorization and a LinearSVC classifier.

## Features

- **High Accuracy**: Achieves 96.65% accuracy on test data
- **Fast Prediction**: Pre-trained model enables instant classification
- **Flexible Input**: Supports single text, file input, and stdin
- **Confidence Scores**: Provides decision scores for prediction confidence

## Project Structure

- `train_simple_model.py` — Script to train and optimize the model
- `predict_fake_news.py` — Script to classify new articles
- `streamlit_app.py` — Streamlit UI for fake news detection
- `run_app.py` — Launches the Streamlit frontend
- `requirements.txt` — Python dependencies
- `dataset/` — Contains news datasets:
  - `Fake.csv` — Fake news dataset
  - `True.csv` — Real news dataset
- `artifacts/` — Contains trained model and vectorizer:
  - `best_model.joblib` — LinearSVC model
  - `tfidf_vectorizer.joblib` — TF-IDF vectorizer

## Usage

### Predicting Fake News

### Streamlit Frontend

This project supports a Streamlit web frontend. To launch the app, run:

```bash
python run_app.py
```
#### Single Text
```bash
python predict_fake_news.py \
  --model ./artifacts/best_model.joblib \
  --vectorizer ./artifacts/tfidf_vectorizer.joblib \
  --text "Your article text goes here"
```

#### From a Text File
```bash
python predict_fake_news.py \
  --model ./artifacts/best_model.joblib \
  --vectorizer ./artifacts/tfidf_vectorizer.joblib \
  --file ./your_articles.txt
```

#### From Standard Input
```bash
echo "Your article text" | python predict_fake_news.py \
  --model ./artifacts/best_model.joblib \
  --vectorizer ./artifacts/tfidf_vectorizer.joblib \
  --stdin
```

### Retraining the Model

To retrain with new data:
```bash
python train_simple_model.py
```
This will:
1. Load `Fake.csv` and `True.csv` datasets
2. Clean and preprocess text data
3. Train a LinearSVC model with GridSearchCV
4. Save the best model and vectorizer to `artifacts/`

## Model Details

- **Algorithm**: LinearSVC (Support Vector Classification)
- **Vectorization**: TF-IDF with bigrams (1–2 word combinations)
- **Preprocessing**: Text cleaning, duplicate removal, exclusive word filtering
- **Performance**: 96.65% accuracy on test set

## Output Format

The prediction script outputs:
- **Label**: FAKE (0) or REAL (1)
- **Decision Score**: Positive = confident REAL, Negative = confident FAKE
- **Text Excerpt**: First 300 characters of input

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- joblib >= 1.3
- streamlit >= 1.28

## Demonstration



