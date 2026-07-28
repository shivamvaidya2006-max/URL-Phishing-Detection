import torch
import pandas as pd
from transformers import DistilBertTokenizer
from src.config import MAX_LEN, MODEL_NAME

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# -------- TRAIN --------
train_df = pd.read_csv("data/processed/train.csv")

train_encodings = tokenizer(
    train_df["url"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)

torch.save(
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": torch.tensor(train_df["label"].values)
    },
    "data/processed/train_tokens.pt"
)

# -------- VALIDATION --------
val_df = pd.read_csv("data/processed/val.csv")

val_encodings = tokenizer(
    val_df["url"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)

torch.save(
    {
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": torch.tensor(val_df["label"].values)
    },
    "data/processed/val_tokens.pt"
)

print("[SUCCESS] Train and validation tokens saved")
