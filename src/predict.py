"""
predict.py
-----------
CPU-optimized inference for phishing URL detection.
"""

import torch
from transformers import DistilBertTokenizer
from src.model import URLPhishingBERT
from src.config import MODEL_NAME, MAX_LEN, MODEL_SAVE_PATH

# Force CPU
DEVICE = torch.device("cpu")

# Load tokenizer once (IMPORTANT)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    model = URLPhishingBERT()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # inference mode
    return model

model = load_model()


def preprocess_url(url: str):
    url = url.lower().strip()
    url = url.replace("http://", "").replace("https://", "")
    return url


def predict_url(url: str):
    url = preprocess_url(url)

    inputs = tokenizer(
        url,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Disable gradients (CRITICAL for speed)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)

    phishing_prob = probs[0][1].item()

    if phishing_prob > 0.6:
        return "PHISHING", phishing_prob
    else:
        return "LEGITIMATE", phishing_prob


if __name__ == "__main__":
    test_url = input("Enter URL to predict: ")
    label, confidence = predict_url(test_url)
    print(f"URL: {test_url}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
