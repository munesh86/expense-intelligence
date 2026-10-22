import pandas as pd
import pytest
from src.data_loader import load_transactions
from src.anomaly_detection import detect_anomalies
from pathlib import Path


from src.expense_analysis import (
    calculate_summary,
    category_wise_expenses,
    monthly_expenses,
)


def test_load_sample_transactions():
    project_dir = Path(__file__).resolve().parent.parent

    csv_path = (
        project_dir
        / "data"
        / "raw"
        / "sample_transactions.csv"
    )

    df = load_transactions(csv_path)

    assert not df.empty
    assert "Date" in df.columns
    assert "Description" in df.columns
    assert "Amount" in df.columns

    assert str(df["Date"].dtype).startswith("datetime64")
    assert df["Amount"].dtype.kind in "if"


def test_expense_summary():
    df = load_sample_data()

    summary = calculate_summary(df)

    assert summary["total_income"] == 90000
    assert summary["total_expenses"] == 15078
    assert summary["net_balance"] == 74922


def test_category_wise_expenses():
    df = load_sample_data()

    result = category_wise_expenses(df)

    assert result["Groceries"] == 3700
    assert result["Transport"] == 2980


def test_monthly_expenses():
    df = load_sample_data()

    result = monthly_expenses(df)

    assert len(result) == 1
    assert result.iloc[0] == 15078


def load_sample_data():
    project_dir = Path(__file__).resolve().parent.parent

    csv_path = (
        project_dir
        / "data"
        / "raw"
        / "sample_transactions.csv"
    )

    return load_transactions(csv_path)
def test_anomaly_detection():
    df = load_sample_data()

    result = detect_anomalies(df, contamination=0.10)

    assert not result.empty
    assert "Anomaly" in result.columns
    assert "Anomaly_Score" in result.columns
    assert (result["Amount"] < 0).all()
    assert result["Anomaly"].dtype == bool

def test_missing_required_columns(tmp_path):
    csv_file = tmp_path / "missing_columns.csv"

    data = pd.DataFrame({
        "Category": ["Groceries"],
        "Method": ["UPI"],
        "Amount": [-500]
    })

    data.to_csv(csv_file, index=False)

    with pytest.raises(
        ValueError,
        match="Unable to identify required transaction columns"
    ):
        load_transactions(csv_file)

def test_duplicate_rows_are_removed(tmp_path):
    csv_file = tmp_path / "duplicates.csv"

    data = pd.DataFrame({
        "Date": ["2026-10-01", "2026-10-01"],
        "Description": ["Grocery Store", "Grocery Store"],
        "Category": ["Groceries", "Groceries"],
        "Method": ["UPI", "UPI"],
        "Amount": [-500, -500]
    })

    data.to_csv(csv_file, index=False)

    result = load_transactions(csv_file)

    assert len(result) == 1


def test_invalid_amount_rows_are_removed(tmp_path):
    csv_file = tmp_path / "invalid_amount.csv"

    data = pd.DataFrame({
        "Date": ["2026-10-01", "2026-10-02"],
        "Description": ["Grocery Store", "Petrol Pump"],
        "Category": ["Groceries", "Transport"],
        "Method": ["UPI", "UPI"],
        "Amount": ["500", "invalid"]
    })

    data.to_csv(csv_file, index=False)

    result = load_transactions(csv_file)

    assert len(result) == 1
    assert result.iloc[0]["Description"] == "Grocery Store"