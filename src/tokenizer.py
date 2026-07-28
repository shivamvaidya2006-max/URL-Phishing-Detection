"""
tokenizer.py
-------------
Tokenizes URLs using a pretrained BERT tokenizer.
Outputs input_ids and attention_mask tensors.
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizer
from src.config import MODEL_NAME, MAX_LEN, TRAIN_DATA_PATH


def load_tokenizer():
    """
    Loads the pretrained BERT tokenizer.
    """
    return DistilBertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_urls(urls, tokenizer):
    """
    Tokenizes a list of URLs into BERT input format.

    Returns:
    - input_ids
    - attention_mask
    """

    encodings = tokenizer(
        urls,
        padding="max_length",     # pad all URLs to MAX_LEN
        truncation=True,          # cut longer URLs
        max_length=MAX_LEN,
        return_tensors="pt"       # return PyTorch tensors
    )

    return encodings


def tokenize_dataset():
    """
    Loads processed CSV and tokenizes URLs.
    """

    print("[INFO] Loading processed dataset...")
    df = pd.read_csv(TRAIN_DATA_PATH)

    urls = df["url"].tolist()
    labels = torch.tensor(df["label"].values)

    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("[INFO] Tokenizing URLs...")
    encodings = tokenize_urls(urls, tokenizer)

    print("[SUCCESS] Tokenization complete!")

    return encodings, labels


if __name__ == "__main__":
    encodings, labels = tokenize_dataset()

    print("input_ids shape:", encodings["input_ids"].shape)
    print("attention_mask shape:", encodings["attention_mask"].shape)
    print("labels shape:", labels.shape)
