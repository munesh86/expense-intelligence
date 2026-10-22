import pandas as pd


def generate_recommendations(df):
    """
    Analyze transactions and generate spending insights.
    Positive Amount = Income
    Negative Amount = Expense
    """

    recommendations = []

    if df.empty:
        return ["Upload transactions to get spending recommendations."]

    # Calculate income and expenses
    income = df.loc[df["Amount"] > 0, "Amount"].sum()

    expenses_df = df[df["Amount"] < 0].copy()
    expenses_df["Expense"] = -expenses_df["Amount"]

    total_expenses = expenses_df["Expense"].sum()

    # Recommendation 1: Income vs expenses
    if income > 0:
        savings = income - total_expenses
        savings_rate = (savings / income) * 100

        if savings < 0:
            recommendations.append(
                "⚠️ Your expenses are higher than your income. "
                "Review your spending and prioritize essential costs."
            )
        elif savings_rate < 20:
            recommendations.append(
                f"💡 Your current savings rate is {savings_rate:.1f}%. "
                "Review non-essential expenses to identify possible savings."
            )
        else:
            recommendations.append(
                f"✅ Your current savings rate is {savings_rate:.1f}%. "
                "Continue monitoring your spending."
            )
    else:
        recommendations.append(
            "ℹ️ No income transactions were found in the uploaded data. "
            "Add income transactions for a savings analysis."
        )

    # Recommendation 2: Highest spending category
    if not expenses_df.empty:
        category_totals = (
            expenses_df.groupby("Category")["Expense"]
            .sum()
            .sort_values(ascending=False)
        )

        if not category_totals.empty:
            highest_category = category_totals.index[0]
            highest_amount = category_totals.iloc[0]

            recommendations.append(
                f"📊 Your highest recorded spending category is "
                f"{highest_category} (₹{highest_amount:,.2f}). "
                "Review these transactions to see whether any savings "
                "opportunities exist."
            )

    # Recommendation 3: Largest transaction
    if not expenses_df.empty:
        largest_expense = expenses_df.loc[
            expenses_df["Expense"].idxmax()
        ]

        recommendations.append(
            f"💸 Your largest recorded expense is "
            f"₹{largest_expense['Expense']:,.2f} for "
            f"'{largest_expense['Description']}'. "
            "Check that the transaction is expected and correctly recorded."
        )

    # Recommendation 4: Transaction count
    if len(expenses_df) > 0:
        recommendations.append(
            f"🧾 You have {len(expenses_df)} expense transactions. "
            "Review them periodically to track your spending habits."
        )

    return recommendations