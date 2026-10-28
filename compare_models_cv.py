from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "training_transactions.csv"
REPORT_DIR = PROJECT_DIR / "reports"


def main():
    print("\n=== 5-Fold Cross-Validation ===\n")

    if not DATA_FILE.exists():
        print(f"Training file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    required = {"Description", "Category"}
    missing = required - set(df.columns)

    if missing:
        print(f"Missing columns: {sorted(missing)}")
        return

    df = df.dropna(subset=["Description", "Category"]).copy()

    df["Description"] = df["Description"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

    df = df[
        (df["Description"] != "")
        & (df["Category"] != "")
    ]

    df = df.drop_duplicates(
        subset=["Description", "Category"]
    )

    category_counts = df["Category"].value_counts()

    if len(category_counts) < 2 or category_counts.min() < 5:
        print("Each category needs at least 5 examples.")
        return

    X = df["Description"]
    y = df["Category"]

    # Both models use the same classifier settings.
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
    )

    # Model 1: Word-level features only
    word_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])

    # Model 2: Word-level + character-level features
    combined_model = Pipeline([
        (
            "features",
            FeatureUnion([
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
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=2.0,
            ),
        ),
    ])

    # Each fold keeps category representation as balanced as possible.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = []

    for model_name, model in [
        ("Word-only", word_model),
        ("Word + character", combined_model),
    ]:
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

        result = {"Model": model_name}

        print(f"\n=== {model_name} ===")

        for metric in scoring:
            values = scores[f"test_{metric}"]

            mean_score = values.mean() * 100
            std_score = values.std() * 100

            result[f"{metric}_mean"] = mean_score
            result[f"{metric}_std"] = std_score

            print(
                f"{metric}: {mean_score:.2f}% "
                f"(fold standard deviation: {std_score:.2f}%)"
            )

        results.append(result)

    # Save comparison
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame(results)

    output.to_csv(
        REPORT_DIR / "cross_validation_comparison.csv",
        index=False,
    )

    print("\nComparison saved to:")
    print(REPORT_DIR / "cross_validation_comparison.csv")


if __name__ == "__main__":
    main()