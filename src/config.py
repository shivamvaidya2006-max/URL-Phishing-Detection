"""
config.py
-----------
Central configuration file for the BERT URL phishing detection project.
Changing values here updates the entire pipeline.
"""

# =============================
# MODEL CONFIGURATION
# =============================

#MODEL_NAME = "bert-base-uncased" #currently not imported for use. Model is explicitly imported in model.py
# You can later switch to: 
MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 64
# URLs are short → 64 tokens is optimal
# Bigger length = slower + no benefit

NUM_LABELS = 2
# 0 = Legitimate
# 1 = Phishing


# =============================
# TRAINING CONFIGURATION
# =============================

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 2e-5
DROPOUT = 0.3
ENCODER_LR_SCALE = 0.25  # encoder learns slower than classifier

# =============================
# PATHS
# =============================

RAW_DATA_PATH = "data/raw/urls.csv"
TRAIN_DATA_PATH = "data/processed/train.csv"
VAL_DATA_PATH = "data/processed/val.csv"
TEST_DATA_PATH = "data/processed/test.csv"

MODEL_SAVE_PATH = "models/bert_url_classifier.pt"
LOG_PATH = "logs/training.log"


# =============================
# SYSTEM CONFIG
# =============================

RANDOM_SEED = 42
DEVICE = "cuda"  # auto-handled later
