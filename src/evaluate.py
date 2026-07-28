import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import URLPhishingBERT
from src.config import MODEL_SAVE_PATH, BATCH_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    data = torch.load("data/processed/val_tokens.pt")

    dataset = TensorDataset(
        data["input_ids"],
        data["attention_mask"],
        data["labels"]
    )

    loader = DataLoader(dataset, batch_size=32)

    model = URLPhishingBERT()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= 0.6).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Accuracy :", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall   :", recall_score(all_labels, all_preds))
    print("F1-score :", f1_score(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
