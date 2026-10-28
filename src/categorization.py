"""Hybrid transaction categorization: clear keyword rules first, ML fallback second."""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

# Clear, common merchant/transaction terms that map reliably to a category.
# Keep category names consistent with the labels used in the dashboard.
CATEGORY_KEYWORDS = {
    "Healthcare": [
        "medicine", "medicines", "pharmacy", "chemist", "medical store",
        "medical shop", "medical bill", "doctor", "clinic", "hospital",
        "healthcare", "health care", "lab test", "blood test", "diagnostic",
        "prescription", "dentist", "dental", "apollo pharmacy", "medplus",
        "netmeds", "pharmeasy", "tata 1mg", "1mg", "pathology lab","optical examination",
"eye examination",
"optical exam",
"optometrist",
    ],
    "Groceries": [
        "grocery", "groceries", "supermarket", "dmart", "d mart",
        "bigbasket", "blinkit", "zepto", "kirana", "provision store",
        "vegetables", "vegetable shop", "fruits", "milk", "dairy",
        "atta", "rice bag", "grocery store","organic food",
        "organic food staples",
    ],
    "Food": [
        "restaurant", "food order", "swiggy", "zomato", "cafe", "coffee shop",
        "pizza", "burger", "dominos", "mcdonald", "kfc", "canteen",
        "bakery", "dining", "takeaway", "food court", "tea stall","lunch order",
        "lunch order payment","office lunch",
"office lunch bill",
    ],
    "Transport": [
        "petrol", "fuel", "diesel", "uber", "ola cab", "rapido", "taxi",
        "cab fare", "auto fare", "metro", "bus fare", "train ticket",
        "railway", "irctc", "redbus", "parking", "toll", "fastag",
        "flight ticket", "transport","bus ticket","commuter pass",
        "transit pass",
        "commuter card",
    ],
    "Utilities": [
        "electricity", "electricity bill", "power bill", "water bill",
        "gas bill", "gas cylinder", "lpg", "mobile recharge", "recharge",
        "broadband", "wifi bill", "internet bill", "telephone bill",
        "jio recharge", "airtel recharge", "postpaid bill","utility service payment",
        "wifi subscription payment",
    ],
    "Entertainment": [
        "movie", "cinema", "movie ticket", "netflix", "hotstar", "prime video",
        "spotify", "concert", "amusement park", "playstation", "xbox",
        "gaming subscription", "game subscription", "theatre ticket","bookmyshow",
        "bookmyshow tickets",
    ],
    "Shopping": [
        "amazon shopping", "flipkart", "myntra", "ajio", "meesho", "clothes",
        "clothing", "shoes", "footwear", "apparel", "fashion store", "shopping",
        "croma", "reliance digital", "electronics store", "mall purchase","shoe shop",
        "shoe shop payment",
    ],
    "Income": [
        "salary", "salary credit", "income", "interest received", "dividend",
        "cash deposit", "salary payment", "pension received", "freelance payment",
    ],
}

def _normalize_description(description):
    """Convert a transaction description to lowercase words with single spaces."""
    if description is None:
        return ""

    text = str(description).strip()
    if text.lower() in {"", "nan", "none", "<na>", "null"}:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_category(description):
    """Return a rule-based category for clear matches, otherwise None."""
    normalized_description = f" {_normalize_description(description)} "
    if not normalized_description.strip():
        return None

    # Healthcare is checked before broader categories so medicine isn't
    # incorrectly classified as entertainment or shopping.
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = _normalize_description(keyword)
            if normalized_keyword and f" {normalized_keyword} " in normalized_description:
                return category

    return None


def predict_category(model, description):
    """Apply reliable keyword rules first, then use the trained model as fallback.

    Examples:
        medicine -> Healthcare
        petrol pump -> Transport
        electricity bill -> Utilities

    Unknown descriptions are sent to the trained classifier when available.
    If neither rules nor a model can classify the description, return
    ``Unclassified``.
    """
    normalized_description = _normalize_description(description)
    if not normalized_description:
        return "Unclassified"

    rule_prediction = _keyword_category(normalized_description)
    if rule_prediction is not None:
        return rule_prediction

    if model is None:
        return "Unclassified"

    try:
        prediction = model.predict([str(description).strip()])
        if len(prediction) > 0:
            predicted_category = str(prediction[0]).strip()
            return predicted_category if predicted_category else "Unclassified"
    except Exception:
        return "Unclassified"

    return "Unclassified"

def train_category_model(training_file):
    """Train a hybrid-feature text classifier for transaction categories."""

    training_file = Path(training_file)

    if not training_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {training_file}"
        )

    data = pd.read_csv(training_file)

    required_columns = {"Description", "Category"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Training CSV is missing columns: {sorted(missing_columns)}"
        )

    # Clean the training data
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
    ]

    # Remove exact duplicate examples
    data = data.drop_duplicates(
        subset=["Description", "Category"]
    )

    if data.empty:
        raise ValueError(
            "The training dataset contains no valid rows."
        )

    if data["Category"].nunique() < 2:
        raise ValueError(
            "At least two different categories are required for training."
        )

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

    # Build the classification pipeline
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

    # Train the complete pipeline
    model.fit(
        data["Description"],
        data["Category"]
    )

    return model