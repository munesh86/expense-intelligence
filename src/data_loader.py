import pandas as pd
from pathlib import Path

# Standard column names used by our project
REQUIRED_COLUMNS = [
    "Date",
    "Description",
    "Category",
    "Method",
    "Amount",
]

# Common bank CSV column names mapped to standard names
COLUMN_ALIASES = {
    "date": "Date",
    "transaction date": "Date",
    "txn date": "Date",
    "value date": "Date",

    "description": "Description",
    "narration": "Description",
    "transaction details": "Description",
    "particulars": "Description",
    "remarks": "Description",

    "category": "Category",
    "expense category": "Category",

    "method": "Method",
    "payment method": "Method",
    "transaction method": "Method",
    "mode": "Method",
    "transaction type": "Method",

    "amount": "Amount",
    "transaction amount": "Amount",
}


def load_transactions(file_source):
    # Read CSV from a file path or uploaded file
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_source}"
            )

        df = pd.read_csv(path)
    else:
        df = pd.read_csv(file_source)

    if df.empty:
        raise ValueError(
            "The CSV file contains no transactions."
        )

    # Normalize column names
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # Rename common bank columns
    rename_map = {
        column: COLUMN_ALIASES[column]
        for column in df.columns
        if column in COLUMN_ALIASES
    }

    df = df.rename(columns=rename_map)

    # Handle Debit and Credit columns
    if "Amount" not in df.columns:
        debit_column = next(
            (
                column for column in df.columns
                if column in ["debit", "withdrawal", "withdrawals"]
            ),
            None,
        )

        credit_column = next(
            (
                column for column in df.columns
                if column in ["credit", "deposit", "deposits"]
            ),
            None,
        )

        if debit_column or credit_column:
            debit = (
                pd.to_numeric(
                    df[debit_column], errors="coerce"
                ).fillna(0)
                if debit_column
                else pd.Series(0, index=df.index)
            )

            credit = (
                pd.to_numeric(
                    df[credit_column], errors="coerce"
                ).fillna(0)
                if credit_column
                else pd.Series(0, index=df.index)
            )

            # Expenses are negative; income is positive
            df["Amount"] = credit.abs() - debit.abs()

    # Check essential columns
    essential_columns = [
        "Date",
        "Description",
        "Amount",
    ]

    missing_columns = [
        column for column in essential_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unable to identify required transaction columns: "
            f"{missing_columns}. "
            "Please check your CSV column names."
        )

    # Add optional columns if missing
    if "Category" not in df.columns:
        df["Category"] = "Unclassified"

    if "Method" not in df.columns:
        df["Method"] = "Unknown"

    # Keep only standard project columns
    df = df[REQUIRED_COLUMNS].copy()

    # Remove exact duplicate rows
    original_count = len(df)
    df = df.drop_duplicates().copy()

    duplicate_count = original_count - len(df)

    if duplicate_count > 0:
        print(
            f"Removed {duplicate_count} duplicate row(s)."
        )

    # Remove rows missing essential fields
    df = df.dropna(
        subset=["Date", "Description", "Amount"]
    ).copy()

    # Validate dates and amounts
    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Amount"] = pd.to_numeric(
        df["Amount"], errors="coerce"
    )

    invalid_rows = (
        df["Date"].isna()
        | df["Amount"].isna()
    )

    invalid_count = int(invalid_rows.sum())

    if invalid_count > 0:
        print(
            f"Warning: Ignoring {invalid_count} row(s) "
            "with invalid dates or amounts."
        )

    df = df.loc[~invalid_rows].copy()

    # Clean descriptions
    df["Description"] = (
        df["Description"].astype(str).str.strip()
    )

    df = df[df["Description"] != ""].copy()

    # Fill missing optional values
    df["Category"] = (
        df["Category"].fillna("Unclassified")
    )

    df["Method"] = (
        df["Method"].fillna("Unknown")
    )

    if df.empty:
        raise ValueError(
            "No valid transactions found in the CSV file."
        )

    return df.reset_index(drop=True)