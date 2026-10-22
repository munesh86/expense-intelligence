from pathlib import Path
import joblib

from src.categorization import train_category_model



PROJECT_DIR = Path(__file__).resolve().parent

TRAINING_FILE = (
    PROJECT_DIR
    / "data"
    / "training_transactions.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "category_model.joblib"


def main():
    print("Training expense categorization model...")

    model = train_category_model(TRAINING_FILE)

    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()