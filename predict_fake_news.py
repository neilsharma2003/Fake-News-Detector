#!/usr/bin/env python3
"""
Predict fake vs real for input text(s) using a saved model and TF-IDF vectorizer.

Usage examples:

  Single text:
    python predict_fake_news.py \
      --model /absolute/path/to/best_model.joblib \
      --vectorizer /absolute/path/to/tfidf_vectorizer.joblib \
      --text "Your article text goes here"

  From a text file (one article per line):
    python predict_fake_news.py \
      --model /absolute/path/to/best_model.joblib \
      --vectorizer /absolute/path/to/tfidf_vectorizer.joblib \
      --file /absolute/path/to/articles.txt

  From stdin:
    echo "Your article text" | python predict_fake_news.py \
      --abs/best_model.joblib --vectorizer /abs/tfidf_vectorizer.joblib --stdin
"""

from __future__ import annotations

import argparse
import sys
import re
import string
from typing import List
from collections import Counter

import joblib


LABEL_MAPPING = {0: "FAKE", 1: "REAL"}


def clean_text(text: str) -> str:
    """Clean text using the same approach as training"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)      # Remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text


def remove_exclusive_and_rare_words(text: str, exclusive_fake: set, exclusive_real: set, rare_threshold: int = 2) -> str:
    """Remove words exclusive to one label and rare tokens"""
    words = text.split()
    words = [w for w in words if w not in exclusive_fake and w not in exclusive_real]
    counts = Counter(words)
    words = [w for w in words if counts[w] >= rare_threshold]
    return " ".join(words)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict fake vs real for input text(s)")
    parser.add_argument("--model", required=True, help="Path to saved model .joblib")
    parser.add_argument("--vectorizer", required=True, help="Path to saved TF-IDF vectorizer .joblib")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text/article to classify")
    group.add_argument("--file", help="Path to a text file with one article per line")
    group.add_argument("--stdin", action="store_true", help="Read a single article from STDIN")
    return parser


def read_inputs(args: argparse.ArgumentParser) -> List[str]:
    if args.text is not None:
        return [args.text.strip()]
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.stdin:
        data = sys.stdin.read().strip()
        return [data] if data else []
    return []


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print("Loading training datasets to replicate exact preprocessing...")
        fake_df = pd.read_csv("./dataset/Fake.csv")
        true_df = pd.read_csv("./dataset/True.csv")

        fake_df['label'] = 0
        true_df['label'] = 1
        
        df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
        df_clean = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        df_clean['clean_text'] = df_clean['text'].apply(clean_text)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df_clean['clean_text'], df_clean['label'], test_size=0.2,
            random_state=42, stratify=df_clean['label']
        )
        
        fake_words = Counter(" ".join(X_train[y_train == 0]).split())
        real_words = Counter(" ".join(X_train[y_train == 1]).split())
        
        exclusive_fake = set(fake_words) - set(real_words)
        exclusive_real = set(real_words) - set(fake_words)
        
        print(f"Computed exclusive words from training set: {len(exclusive_fake)} fake, {len(exclusive_real)} real")
        
    except Exception as e:
        print(f"Error: Could not load training data for preprocessing: {e}")
        print("Please ensure Fake.csv and True.csv are in the current directory.")
        sys.exit(1)

    texts = read_inputs(args)
    if not texts:
        print("No input text provided.")
        sys.exit(1)

    processed_texts = []
    for text in texts:
        cleaned = clean_text(text)
        
        processed = remove_exclusive_and_rare_words(cleaned, exclusive_fake, exclusive_real, rare_threshold=2)
        processed_texts.append(processed)
        
        print(f"\nPreprocessing steps:")
        print(f"Original: {text[:100]}...")
        print(f"Cleaned: {cleaned[:100]}...")
        print(f"Processed (after exclusive word removal): {processed[:100]}...")
        print("-" * 50)

    X = vectorizer.transform(processed_texts)

    preds = model.predict(X)

    scores = None
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
        except Exception:
            pass

    for idx, (text, processed_text) in enumerate(zip(texts, processed_texts)):
        label = LABEL_MAPPING.get(int(preds[idx]), str(preds[idx]))
        header = f"Example {idx + 1}: {label}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        if scores is not None:
            try:
                print(f"Decision score: {float(scores[idx]):.4f}")
            except Exception:
                pass
        
        print("Original text excerpt:")
        excerpt = (text[:300] + "…") if len(text) > 300 else text
        print(excerpt)
        print()
        
        print("Processed text excerpt:")
        processed_excerpt = (processed_text[:300] + "…") if len(processed_text) > 300 else processed_text
        print(processed_excerpt)
        print()


if __name__ == "__main__":
    main()


