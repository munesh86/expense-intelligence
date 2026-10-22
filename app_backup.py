import streamlit as st
import plotly.express as px
import joblib
from src.recommendations import generate_recommendations
import pandas as pd

from src.categorization import predict_category
from pathlib import Path

from src.data_loader import load_transactions
from src.expense_analysis import (
    calculate_summary,
    category_wise_expenses,
    monthly_expenses,
)


# Configure the dashboard
st.set_page_config(
    page_title="Personal Expense Intelligence",
    page_icon="💰",
    layout="wide"
)
# Professional dashboard styling
st.markdown(
    """
    <style>
        /* Main application */
        .stApp {
            background-color: #F7F9FC;
            color: #1F2937;
        }

        /* Main content layout */
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1, h2, h3 {
            color: #172B4D;
            font-family: Arial, sans-serif;
            font-weight: 600;
            letter-spacing: -0.3px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px;
        }

        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 14px;
        }

        [data-testid="stMetricValue"] {
            color: #172B4D;
            font-size: 28px;
            font-weight: 600;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #263449;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            font-weight: 500;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 8px;
        }

        /* Dividers */
        hr {
            border-color: #E2E8F0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard title
st.title("💰 AI-Powered Personal Expense Intelligence System")

st.markdown(
    "Analyze your income, expenses, and spending patterns."
)


# Load sample transaction data
project_dir = Path(__file__).resolve().parent
model_path = project_dir / "models" / "category_model.joblib"

if model_path.exists():
    category_model = joblib.load(model_path)
else:
    category_model = None

st.sidebar.header("📂 Upload Transactions")

uploaded_file = st.sidebar.file_uploader(
    "Upload your transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = load_transactions(uploaded_file)

        # Predict categories for unclassified transactions
        if category_model is not None:
            mask = (
                df["Category"].isna()
                | df["Category"].astype(str).str.strip().str.lower().isin(
                    ["unclassified", "", "unknown"]
                )
            )

            if mask.any():
                descriptions = df.loc[mask, "Description"].astype(str)
                df.loc[mask, "Category"] = category_model.predict(
                    descriptions
                )

        st.sidebar.success("CSV uploaded successfully!")
        st.sidebar.info(f"Transactions loaded: {len(df)}")

    except Exception as error:
        st.error(f"Unable to load transactions: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unable to load transactions: {error}")
        st.stop()

else:
    csv_path = (
        project_dir
        / "data"
        / "raw"
        / "sample_transactions.csv"
    )

    try:
        df = load_transactions(csv_path)

    except Exception as error:
        st.error(f"Unable to load transactions: {error}")
        st.stop()


# Sidebar filters
st.sidebar.header("Dashboard Filters")

categories = sorted(df["Category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select categories",
    options=categories,
    default=categories
)

filtered_df = df[df["Category"].isin(selected_categories)]


# Calculate financial summary
summary = calculate_summary(filtered_df)


# Display key financial metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Income",
    f"₹{summary['total_income']:,.2f}"
)

col2.metric(
    "Total Expenses",
    f"₹{summary['total_expenses']:,.2f}"
)

col3.metric(
    "Net Cash Flow",
    f"₹{summary['net_balance']:,.2f}"
)

col4.metric(
    "Transactions",
    summary["transaction_count"]
)


st.divider()


# Category-wise expense chart
st.subheader("📊 Expenses by Category")

category_data = category_wise_expenses(filtered_df)

if not category_data.empty:
    category_df = category_data.rename("Expense").reset_index()

    fig_category = px.pie(
        category_df,
        names="Category",
        values="Expense",
        title="Expense Distribution",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_category.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color="#263449",
            size=13
        ),
        title=dict(
            text="Expense Distribution",
            x=0.02,
            xanchor="left"
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=20, r=120, t=60, b=20)
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True,
        config={"displayModeBar": False}
    )
    

else:
    st.info("No expenses found for the selected categories.")


# Monthly expense chart
# Monthly Spending Trend
st.subheader("📅 Monthly Spending Trend")

monthly_df = df[df["Amount"] < 0].copy()

if not monthly_df.empty:
    monthly_df["Month"] = (
        monthly_df["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_df["Expense"] = -monthly_df["Amount"]

    monthly_summary = (
        monthly_df.groupby("Month", as_index=False)["Expense"]
        .sum()
        .sort_values("Month")
    )

    fig_monthly = px.bar(
    monthly_summary,
    x="Month",
    y="Expense",
    title="Monthly Expenses",
    labels={
        "Month": "Month",
        "Expense": "Expenses (₹)"
    },
    text_auto=".0f",
    color_discrete_sequence=["#3978C5"]
)

    fig_monthly.update_layout(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Arial, sans-serif",
        color="#263449",
        size=13
    ),
    title=dict(
        text="Monthly Expenses",
        x=0.02,
        xanchor="left"
    ),
    showlegend=False,
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis_title="Month",
    yaxis_title="Expenses (₹)"
)

    fig_monthly.update_xaxes(
    showgrid=False
)

    fig_monthly.update_yaxes(
    gridcolor="#E8EDF4",
    zeroline=False
)

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Expenses (₹)",
        xaxis={
            "type": "category",
            "tickformat": None
        }
    )

    st.plotly_chart(fig_monthly, use_container_width=True)

else:
    st.info("No expense transactions available for the monthly chart.")

# Top 5 Highest Expenses
st.subheader("💸 Top 5 Highest Expenses")

expense_df = df[df["Amount"] < 0].copy()

if not expense_df.empty:
    # Convert negative amounts into positive expense values
    expense_df["Expense Amount"] = -expense_df["Amount"]

    # Select the 5 largest expenses
    top_expenses = expense_df.nlargest(
        5, "Expense Amount"
    ).copy()

    # Format dates for display
    top_expenses["Date"] = (
        top_expenses["Date"].dt.strftime("%d %b %Y")
    )

    # Display selected columns
    top_expenses = top_expenses[
        ["Date", "Description", "Category", "Expense Amount"]
    ]

    top_expenses = top_expenses.rename(
        columns={
            "Expense Amount": "Amount (₹)"
        }
    )

    st.dataframe(
        top_expenses,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No expense transactions available.")

# Spending Insights & Recommendations
st.subheader("Spending Insights")

try:
    recommendations = generate_recommendations(filtered_df)

    if recommendations:
        for recommendation in recommendations:
            st.markdown(
                f"""
                <div style="
                    background-color: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-left: 4px solid #3978C5;
                    border-radius: 6px;
                    padding: 14px 18px;
                    margin-bottom: 10px;
                    color: #263449;
                    font-size: 15px;
                ">
                    {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("No recommendations available.")

except Exception as error:
    st.warning(
        f"Unable to generate spending recommendations: {error}"
    )
# Display transaction table
st.subheader("📋 Transaction History")

# Create a copy of the filtered transactions
display_df = filtered_df.copy()

# Search and filter options
col1, col2, col3 = st.columns(3)

with col1:
    search_text = st.text_input(
        "🔍 Search Description",
        placeholder="e.g. Grocery Store"
    )

with col2:
    categories = sorted(
        display_df["Category"]
        .fillna("Unclassified")
        .astype(str)
        .unique()
    )

    selected_category = st.selectbox(
        "🏷️ Filter by Category",
        ["All Categories"] + categories
    )

with col3:
    transaction_type = st.selectbox(
        "💰 Transaction Type",
        ["All Transactions", "Expenses", "Income"]
    )

# Apply description search
if search_text:
    display_df = display_df[
        display_df["Description"]
        .astype(str)
        .str.contains(search_text, case=False, na=False)
    ]

# Apply category filter
if selected_category != "All Categories":
    display_df = display_df[
        display_df["Category"].fillna("Unclassified").astype(str)
        == selected_category
    ]

# Apply income/expense filter
if transaction_type == "Expenses":
    display_df = display_df[display_df["Amount"] < 0]

elif transaction_type == "Income":
    display_df = display_df[display_df["Amount"] > 0]

# Display transaction count
st.caption(f"Showing {len(display_df)} transactions")

# Format dates for display
if not display_df.empty:
    display_df["Date"] = pd.to_datetime(
        display_df["Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Download filtered transactions
    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Transactions",
        data=csv_data,
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("No transactions match your selected filters.")

st.caption(
    "Prototype version — using sample transactions for development."
)
st.divider()

st.subheader("🤖 AI Expense Category Predictor")

st.write(
    "Enter a transaction description to predict its expense category."
)

if category_model is not None:

    description = st.text_input(
        "Transaction Description",
        placeholder="Example: DMart Grocery"
    )

    if st.button("Predict Category"):

        if description.strip():

            predicted_category = predict_category(
                category_model,
                description
            )

            st.success(
                f"Predicted Category: {predicted_category}"
            )

        else:
            st.warning("Please enter a transaction description.")

else:
    st.warning(
        "Model not found. Please run python train_models.py first."
    )