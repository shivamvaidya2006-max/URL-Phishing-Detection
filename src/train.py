"""
train.py
---------
Training loop for BERT-based URL phishing detection.
"""

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    ENCODER_LR_SCALE
)

from src.dataset import URLDataset
from src.model import URLPhishingBERT

torch.backends.cudnn.benchmark = True

def train():
    # -----------------------------
    # DEVICE SETUP (CPU / GPU)
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------------------
# LOAD PRE-TOKENIZED DATA
# -----------------------------
    dataset = URLDataset("data/processed/train_tokens.pt")


    dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,      # try 4 → 6 → 8
    pin_memory=True
)

    # -----------------------------
    # MODEL SETUP
    # -----------------------------
    model = URLPhishingBERT()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"[INFO] Trainable params: {trainable:,} / {total:,}")

    # -----------------------------
    # LOSS & OPTIMIZER
    # -----------------------------
    criterion = CrossEntropyLoss(label_smoothing=0.05)

    optimizer = AdamW(
    [
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": LEARNING_RATE,
        },
        {
            "params": [
                p
                for p in model.bert.transformer.layer[-2:].parameters()
                if p.requires_grad
            ],
            "lr": LEARNING_RATE * ENCODER_LR_SCALE,
        },
    ]
)


    scaler = GradScaler()

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        print(f"\n[INFO] Epoch {epoch + 1}/{EPOCHS}")

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)


            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
