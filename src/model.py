"""
model.py
---------
BERT-based model for URL phishing detection.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel
from src.config import MODEL_NAME, NUM_LABELS, DROPOUT


class URLPhishingBERT(nn.Module):
    """
    BERT + classification head for URL phishing detection.
    """

    def __init__(self):
        super(URLPhishingBERT, self).__init__()

        # Load pretrained BERT encoder
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        # -----------------------------
        # FREEZE / UNFREEZE DISTILBERT LAYERS
        # -----------------------------

        # Freeze all layers first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        UNFREEZE_LAST_N = 2  # unfreeze top 2 layers

        for layer in self.bert.transformer.layer[-UNFREEZE_LAST_N:]:
            for param in layer.parameters():
                param.requires_grad = True


        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(DROPOUT)

        # Classification head
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            NUM_LABELS
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        input_ids: [batch_size, max_len]
        attention_mask: [batch_size, max_len]
        """

        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract [CLS] token representation
        # Shape: [batch_size, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)

        # Final classification
        logits = self.classifier(cls_embedding)

        return logits
