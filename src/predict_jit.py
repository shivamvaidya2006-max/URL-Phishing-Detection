import torch
from transformers import DistilBertTokenizer
from src.config import MODEL_NAME, MAX_LEN

DEVICE = torch.device("cpu")

# Load tokenizer (must match training tokenizer)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Load TorchScript model
model = torch.jit.load("models/url_phishing_jit.pt")
model.eval()


def preprocess_url(url: str):
    return url.lower().replace("http://", "").replace("https://", "").strip()


def predict_url(url: str):
    url = preprocess_url(url)

    inputs = tokenizer(
        url,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 🔒 Inference mode
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)

    legit_prob = probs[0][0].item()
    phish_prob = probs[0][1].item()

    label = "PHISHING" if phish_prob >= 0.6 else "LEGITIMATE"

    return label, legit_prob, phish_prob


if __name__ == "__main__":
    url = input("Enter URL to predict: ")
    label, legit_prob, phish_prob = predict_url(url)

    print(f"\nPrediction: {label}")
    print(f"Legitimate probability: {legit_prob:.4f}")
    print(f"Phishing probability: {phish_prob:.4f}")

