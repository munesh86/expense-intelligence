from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.categorization import predict_category


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_FILE = PROJECT_DIR / "data" / "training_transactions.csv"
TEST_FILE = PROJECT_DIR / "data" / "independent_test_transactions.csv"
REPORT_DIR = PROJECT_DIR / "reports"


def load_dataset(file_path):
    """Load and validate a transaction categorization CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = {"Description", "Category"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = data.dropna(
        subset=["Description", "Category"]
    ).copy()

    data["Description"] = (
        data["Description"].astype(str).str.strip()
    )
    data["Category"] = (
        data["Category"].astype(str).str.strip()
    )

    data = data[
        (data["Description"] != "")
        & (data["Category"] != "")
    ].copy()

    data = data.drop_duplicates(
        subset=["Description", "Category"]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            f"{file_path.name} has no valid examples."
        )

    return data


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    file_path,
    title,
):
    """Save a confusion matrix image."""

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        values_format="d",
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(title)
    fig.tight_layout()

    fig.savefig(
        file_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    print("\n=== Unseen Data Model Evaluation ===\n")

    try:
        train_df = load_dataset(TRAIN_FILE)
        test_df = load_dataset(TEST_FILE)

    except (FileNotFoundError, ValueError) as error:
        print(f"Dataset error: {error}")
        return

    # Check for exact description overlap between training and testing.
    train_descriptions = set(
        train_df["Description"].str.casefold().str.strip()
    )
    test_descriptions = set(
        test_df["Description"].str.casefold().str.strip()
    )

    overlapping = train_descriptions & test_descriptions

    if overlapping:
        print(
            "Evaluation stopped: some descriptions appear in "
            "both training and testing files."
        )
        print("Overlapping descriptions:")
        for description in sorted(overlapping):
            print(f"- {description}")
        return

    train_categories = set(train_df["Category"])
    test_categories = set(test_df["Category"])

    # Each tested category must be represented in the training data.
    missing_categories = test_categories - train_categories

    if missing_categories:
        print(
            "Evaluation stopped. These test categories are "
            "missing from the training data:"
        )
        print(sorted(missing_categories))
        return

    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")

    print("\nTraining examples per category:")
    print(train_df["Category"].value_counts().to_string())

    print("\nTesting examples per category:")
    print(test_df["Category"].value_counts().to_string())

    X_train = train_df["Description"]
    y_train = train_df["Category"]

    X_test = test_df["Description"]
    y_test = test_df["Category"]

    # Train a classifier using only the training CSV.
        # Combine word-level and character-level text features
    text_features = FeatureUnion([
        (
            "word_features",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                sublinear_tf=True,
            ),
        ),
    ])

    # Use the same configuration as the training function
    model = Pipeline([
        ("features", text_features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=2.0,
            ),
        ),
    ])

    # Train the model
    model.fit(X_train, y_train)
    # Evaluate the ML model alone.
    ml_predictions = model.predict(X_test)

    # Evaluate keyword rules + ML fallback.
    hybrid_predictions = [
        predict_category(model, description)
        for description in X_test
    ]

    ml_accuracy = accuracy_score(
        y_test,
        ml_predictions,
    )

    hybrid_accuracy = accuracy_score(
        y_test,
        hybrid_predictions,
    )

    print("\n=== Model Comparison ===")
    print(f"ML-only accuracy: {ml_accuracy * 100:.2f}%")
    print(f"Hybrid accuracy:  {hybrid_accuracy * 100:.2f}%")

    labels = sorted(train_categories)

    # Compare individual predictions.
    comparison = pd.DataFrame({
        "Description": X_test.to_numpy(),
        "Actual Category": y_test.to_numpy(),
        "ML Prediction": ml_predictions,
        "Hybrid Prediction": hybrid_predictions,
    })

    comparison["ML Correct"] = (
        comparison["Actual Category"]
        == comparison["ML Prediction"]
    )

    comparison["Hybrid Correct"] = (
        comparison["Actual Category"]
        == comparison["Hybrid Prediction"]
    )

    print("\n=== Individual Predictions ===")
    print(comparison.to_string(index=False))

    incorrect = comparison[
        ~comparison["Hybrid Correct"]
    ].copy()

    print("\n=== Incorrect Hybrid Predictions ===")

    if incorrect.empty:
        print("No incorrect hybrid predictions.")
    else:
        print(
            incorrect[
                [
                    "Description",
                    "Actual Category",
                    "Hybrid Prediction",
                ]
            ].to_string(index=False)
        )

    # Print detailed metrics for both approaches.
    print("\n=== ML-only Classification Report ===")

    print(
        classification_report(
            y_test,
            ml_predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("\n=== Hybrid Classification Report ===")

    print(
        classification_report(
            y_test,
            hybrid_predictions,
            labels=labels,
            zero_division=0,
        )
    )

    # Save all reports.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        REPORT_DIR / "individual_predictions.csv",
        index=False,
    )

    incorrect.to_csv(
        REPORT_DIR / "incorrect_predictions.csv",
        index=False,
    )

    for name, predictions in [
        ("ml", ml_predictions),
        ("hybrid", hybrid_predictions),
    ]:
        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).transpose().to_csv(
            REPORT_DIR / f"{name}_classification_report.csv"
        )

        save_confusion_matrix(
            y_test,
            predictions,
            labels,
            REPORT_DIR / f"{name}_confusion_matrix.png",
            f"{name.upper()} Confusion Matrix",
        )

    print("\nReports saved in:", REPORT_DIR)
    print("- individual_predictions.csv")
    print("- incorrect_predictions.csv")
    print("- ml_classification_report.csv")
    print("- hybrid_classification_report.csv")
    print("- ml_confusion_matrix.png")
    print("- hybrid_confusion_matrix.png")

    print(
        "\nNote: This is a small synthetic test dataset. "
        "Results are preliminary and may not represent "
        "performance on real bank statements."
    )


if __name__ == "__main__":
    main()