#!/usr/bin/env python3
"""
Simplified Fake News Detection Training Script
Based on the working notebook approach that achieves 94.84% accuracy
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def clean_text(text):
    """Clean text using the same approach as the notebook"""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)      # Remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text

def remove_exclusive_and_rare_words(text, exclusive_fake, exclusive_real, rare_threshold=2):
    """Remove words exclusive to one label and rare tokens"""
    words = text.split()
    words = [w for w in words if w not in exclusive_fake and w not in exclusive_real]
    counts = Counter(words)
    words = [w for w in words if counts[w] >= rare_threshold]
    return " ".join(words)

def main():
    print("Loading datasets...")
    fake_df = pd.read_csv("dataset/Fake.csv")
    true_df = pd.read_csv("dataset/True.csv")

    print(f"Fake dataset shape: {fake_df.shape}")
    print(f"True dataset shape: {true_df.shape}")
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    
    before_dedup = len(df)
    df_clean = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    after_dedup = len(df_clean)
    print(f"Removed duplicates: {before_dedup - after_dedup} (remaining: {after_dedup})")
    
    print(f"\nLabel distribution:")
    print(f"Fake (0): {len(df_clean[df_clean['label'] == 0])}")
    print(f"Real (1): {len(df_clean[df_clean['label'] == 1])}")
    
    print("\nCleaning text...")
    df_clean['clean_text'] = df_clean['text'].apply(clean_text)
    
    print("Identifying exclusive words...")
    fake_words = Counter(" ".join(df_clean[df_clean['label'] == 0]['clean_text']).split())
    real_words = Counter(" ".join(df_clean[df_clean['label'] == 1]['clean_text']).split())
    
    exclusive_fake = set(fake_words) - set(real_words)
    exclusive_real = set(real_words) - set(fake_words)
    
    print(f"Exclusive fake words: {len(exclusive_fake)}")
    print(f"Exclusive real words: {len(exclusive_real)}")
    
    print("Removing exclusive and rare words...")
    df_clean['clean_text'] = df_clean['clean_text'].apply(
        lambda x: remove_exclusive_and_rare_words(x, exclusive_fake, exclusive_real, rare_threshold=2)
    )
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['clean_text'], df_clean['label'], test_size=0.2,
        random_state=42, stratify=df_clean['label']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=3, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Training features: {X_train_tfidf.shape}")
    print(f"Test features: {X_test_tfidf.shape}")
    
    print("\nTraining LinearSVC...")
    base_model = LinearSVC(max_iter=5000)
    base_model.fit(X_train_tfidf, y_train)
    
    y_pred_base = base_model.predict(X_test_tfidf)
    base_acc = accuracy_score(y_test, y_pred_base)
    print(f"\nBase LinearSVC Accuracy: {base_acc:.4f}")
    print(classification_report(y_test, y_pred_base))
    
    print("\nRunning GridSearchCV...")
    param_grid = {
        'C': np.logspace(-3, 2, 6),  # [0.001, 0.01, 0.1, 1, 10, 100]
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        estimator=LinearSVC(max_iter=5000),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test_tfidf, y_test)
    print(f"Test Accuracy with best params: {test_acc:.4f}")
    
    print("\nSaving model and vectorizer...")
    os.makedirs('./artifacts', exist_ok=True)
    
    model_path = './artifacts/best_model.joblib'
    vectorizer_path = './artifacts/tfidf_vectorizer.joblib'
    
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    
    print("\nTesting predictions on sample texts...")
    test_texts = [
        "websites have reported that two British army officers were captured by Russian special forces during a raid into Ukraine",
        "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a fiscal conservative on Sunday and urged budget restraint in 2018."
    ]
    
    for i, text in enumerate(test_texts):
        cleaned = clean_text(text)
        cleaned = remove_exclusive_and_rare_words(cleaned, exclusive_fake, exclusive_real, rare_threshold=2)
        
        X_sample = vectorizer.transform([cleaned])
        prediction = best_model.predict(X_sample)[0]
        decision_score = best_model.decision_function(X_sample)[0]
        
        label = "FAKE" if prediction == 0 else "REAL"
        print(f"\nText {i+1}: {label} (score: {decision_score:.4f})")
        print(f"Excerpt: {text[:100]}...")

if __name__ == "__main__":
    main()
