import pandas as pd


def calculate_summary(df):
    """Calculate overall income, expenses, and balance."""

    income = df.loc[df["Amount"] > 0, "Amount"].sum()

    expenses = -df.loc[df["Amount"] < 0, "Amount"].sum()

    balance = income - expenses

    return {
        "total_income": round(float(income), 2),
        "total_expenses": round(float(expenses), 2),
        "net_balance": round(float(balance), 2),
        "transaction_count": len(df)
    }


def category_wise_expenses(df):
    """Calculate expenses for each category."""

    expenses = df[df["Amount"] < 0].copy()

    expenses["Expense"] = -expenses["Amount"]

    result = (
        expenses.groupby("Category")["Expense"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def monthly_expenses(df):
    """Calculate total expenses for each month."""

    expenses = df[df["Amount"] < 0].copy()

    expenses["Month"] = expenses["Date"].dt.to_period("M")

    expenses["Expense"] = -expenses["Amount"]

    result = (
        expenses.groupby("Month")["Expense"]
        .sum()
        .sort_index()
    )

    return result