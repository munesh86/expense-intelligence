import pandas as pd

from src.recommendations import generate_recommendations


def test_generate_recommendations():
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-10-01",
            "2026-10-02",
            "2026-10-03"
        ]),
        "Description": [
            "Salary Credit",
            "Grocery Store",
            "Petrol Pump"
        ],
        "Category": [
            "Income",
            "Groceries",
            "Transport"
        ],
        "Method": [
            "Bank Transfer",
            "UPI",
            "UPI"
        ],
        "Amount": [
            50000,
            -3000,
            -2000
        ]
    })

    recommendations = generate_recommendations(df)

    assert len(recommendations) >= 3
    assert any(
        "savings rate" in item.lower()
        for item in recommendations
    )
    assert any(
        "groceries" in item.lower()
        for item in recommendations
    )


def test_empty_transactions():
    df = pd.DataFrame(
        columns=[
            "Date",
            "Description",
            "Category",
            "Method",
            "Amount"
        ]
    )

    recommendations = generate_recommendations(df)

    assert len(recommendations) == 1
    assert "Upload transactions" in recommendations[0]