import torch
from src.model import URLPhishingBERT
from src.config import MODEL_SAVE_PATH

DEVICE = torch.device("cpu")

model = URLPhishingBERT()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

example_input_ids = torch.randint(0, 30522, (1, 64))
example_attention_mask = torch.ones((1, 64), dtype=torch.long)

scripted_model = torch.jit.trace(
    model,
    (example_input_ids, example_attention_mask)
)

scripted_model.save("models/url_phishing_jit.pt")
print("[SUCCESS] TorchScript model saved")
