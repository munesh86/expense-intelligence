import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.05):
    """
    Flag potentially unusual expense transactions.

    Parameters:
        df: DataFrame containing Date, Description, Category,
            Method, and Amount.
        contamination: Expected proportion of unusual records.
            This is a modeling assumption, not a fraud rate.

    Returns:
        A copy of the expense records with anomaly flags and scores.
    """

    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")

    # Consider expense transactions only
    expenses = df[df["Amount"] < 0].copy()

    if len(expenses) < 5:
        raise ValueError(
            "At least 5 expense transactions are required "
            "for this prototype."
        )

    # Convert negative expense amounts to positive values
    expenses["Expense_Amount"] = -expenses["Amount"]

    # Train the anomaly detection model
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )

    features = expenses[["Expense_Amount"]]
    predictions = model.fit_predict(features)

    # -1 means potentially unusual; 1 means typical
    expenses["Anomaly"] = predictions == -1

    # Higher scores indicate more typical records
    expenses["Anomaly_Score"] = model.decision_function(features)

    return expenses