import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/url.csv")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_url(url):
    return (
        url.lower()
        .replace("http://", "")
        .replace("https://", "")
        .strip()
    )

def main():
    df = pd.read_csv(RAW_PATH)

    # Basic validation
    df = df.dropna()
    df = df[df["url"].str.len() > 3]
    df["label"] = df["label"].astype(int)

    # Clean URLs
    df["url"] = df["url"].apply(clean_url)

    # Train / validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    print("[SUCCESS] train.csv and val.csv created")

if __name__ == "__main__":
    main()
