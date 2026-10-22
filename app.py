import html
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomaly_detection import detect_anomalies
from src.categorization import predict_category
from src.data_loader import load_transactions
from src.expense_analysis import calculate_summary, category_wise_expenses
from src.recommendations import generate_recommendations


# Page setup must be the first Streamlit command.
st.set_page_config(
    page_title="Expense Intelligence",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained, consistent style. The .streamlit/config.toml file included
# with this update also sets Streamlit's native widgets and dataframes to light.
st.markdown(
    """
    <style>
        .stApp {
            background: #F6F8FB;
            color: #172B4D;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #172B4D;
            letter-spacing: -0.35px;
            font-weight: 650;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E3E8F0;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E3E8F0;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
        }

        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 0.88rem;
        }

        [data-testid="stMetricValue"] {
            color: #172B4D;
            font-size: 1.7rem;
            font-weight: 650;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #E3E8F0;
            border-radius: 8px;
            overflow: hidden;
        }

        .section-note {
            color: #64748B;
            font-size: 0.9rem;
            margin-top: -0.5rem;
        }

        .insight-card {
            background: #FFFFFF;
            border: 1px solid #E3E8F0;
            border-left: 4px solid #3978C5;
            border-radius: 7px;
            padding: 13px 16px;
            margin: 0 0 10px 0;
            color: #263449;
            font-size: 0.94rem;
            line-height: 1.5;
        }

        div.stButton > button, div.stDownloadButton > button {
            border-radius: 7px;
            font-weight: 550;
        }

        hr {
            border-color: #E3E8F0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Project paths and trained model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
SAMPLE_CSV_PATH = PROJECT_DIR / "data" / "raw" / "sample_transactions.csv"

if MODEL_PATH.exists():
    try:
        category_model = joblib.load(MODEL_PATH)
    except Exception as error:
        category_model = None
        st.sidebar.warning(f"Category model could not be loaded: {error}")
else:
    category_model = None


# Sidebar: upload, data source, filters, and navigation
st.sidebar.title("Expense Intelligence")
st.sidebar.caption("Personal finance overview")
st.sidebar.divider()
st.sidebar.subheader("Transaction data")

uploaded_file = st.sidebar.file_uploader(
    "Upload bank transaction CSV",
    type=["csv"],
    help="Use a CSV containing transaction dates, descriptions, and amounts.",
)

if uploaded_file is not None:
    try:
        df = load_transactions(uploaded_file)
        data_source = uploaded_file.name
        using_sample_data = False
    except Exception as error:
        st.error(f"Unable to load the uploaded CSV: {error}")
        st.stop()
else:
    try:
        df = load_transactions(SAMPLE_CSV_PATH)
        data_source = "Sample transactions"
        using_sample_data = True
    except Exception as error:
        st.error(f"Unable to load the sample transactions: {error}")
        st.stop()

# Ensure predictable category values for filtering and display.
df = df.copy()
df["Category"] = df["Category"].fillna("Unclassified").astype(str).str.strip()
df.loc[df["Category"] == "", "Category"] = "Unclassified"
df["Description"] = df["Description"].fillna("").astype(str)
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Amount"]).copy()

# Predict only rows that do not already have a useful category.
# Automatically categorize unclassified transactions
if not df.empty:
    uncategorized_mask = (
        df["Category"]
        .fillna("Unclassified")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["unclassified", "", "unknown", "nan"])
    )

    if uncategorized_mask.any():
        try:
            df.loc[uncategorized_mask, "Category"] = [
                predict_category(category_model, description)
                for description in df.loc[
                    uncategorized_mask, "Description"
                ].fillna("").astype(str)
            ]

        except Exception as error:
            st.sidebar.warning(
                f"Automatic categorization failed: {error}"
            )

st.sidebar.success(f"{len(df):,} transactions loaded")
st.sidebar.caption(f"Data source: {data_source}")
if using_sample_data:
    st.sidebar.caption("Sample data is shown until you upload a CSV.")

st.sidebar.divider()
st.sidebar.subheader("Dashboard filters")
all_categories = sorted(df["Category"].dropna().astype(str).unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Categories",
    options=all_categories,
    default=all_categories,
    help="These selections affect the overview and its insights.",
)
filtered_df = df[df["Category"].isin(selected_categories)].copy()

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    options=["Overview", "Transactions", "Category Predictor"],
    index=0,
)
st.sidebar.caption("Personal Expense Intelligence System")


# Common page heading
st.title("Expense Intelligence")
st.markdown(
    "A clear view of your cash flow, spending patterns, and transaction history."
)
status_col1, status_col2 = st.columns([2, 1])
with status_col1:
    st.caption(f"Data source: **{data_source}**")
with status_col2:
    st.caption(f"Last loaded: {len(df):,} transactions")


# Plotly layout shared by both charts
def style_chart(figure, title):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color="#263449",
            size=12
        ),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(
                size=17,
                color="#172B4D"
            )
        ),
        legend=dict(
            font=dict(color="#263449")
        ),
        margin=dict(l=16, r=16, t=58, b=16)
    )

    figure.update_xaxes(
        title_font=dict(color="#263449"),
        tickfont=dict(color="#64748B")
    )

    figure.update_yaxes(
        title_font=dict(color="#263449"),
        tickfont=dict(color="#64748B"),
        gridcolor="#E8EDF4",
        zeroline=False
    )

    return figure

if page == "Overview":
    st.subheader("Financial overview")
    st.markdown('<p class="section-note">Summary of the transactions in your selected categories.</p>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No transactions match the selected categories. Select one or more categories in the sidebar.")
    else:
        summary = calculate_summary(filtered_df)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total income", f"₹{summary['total_income']:,.2f}")
        metric_cols[1].metric("Total expenses", f"₹{summary['total_expenses']:,.2f}")
        metric_cols[2].metric("Net cash flow", f"₹{summary['net_balance']:,.2f}")
        metric_cols[3].metric("Transactions", f"{summary['transaction_count']:,}")

        st.divider()

        chart_left, chart_right = st.columns([1.15, 1])
        with chart_left:
            st.subheader("Monthly spending")
            monthly_df = filtered_df[filtered_df["Amount"] < 0].copy()
            if not monthly_df.empty:
                monthly_df["Month"] = monthly_df["Date"].dt.to_period("M").astype(str)
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
                    text="Expense",
                    color_discrete_sequence=["#3978C5"],
                    labels={"Month": "Month", "Expense": "Expenses (₹)"},
                )
                style_chart(fig_monthly, "Monthly expenses")
                fig_monthly.update_traces(
                    texttemplate="₹%{y:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    marker_line_width=0,
                )
                fig_monthly.update_layout(showlegend=False)
                fig_monthly.update_xaxes(
    type="category",
    showgrid=False,
    tickfont={"color": "#64748B"},
)
                fig_monthly.update_yaxes(
                    gridcolor="#E8EDF4",
                    zeroline=False,
                    tickprefix="₹",
                    tickfont={"color": "#64748B"},
                )
                st.plotly_chart(fig_monthly, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No expense transactions are available for the monthly chart.")

        with chart_right:
            st.subheader("Spending by category")
            category_data = category_wise_expenses(filtered_df)
            if category_data is not None and not category_data.empty:
                category_df = category_data.rename("Expense").reset_index()
                fig_category = px.pie(
                    category_df,
                    names="Category",
                    values="Expense",
                    hole=0.48,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                style_chart(fig_category, "Expense distribution")
                fig_category.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    hovertemplate="%{label}<br>Expenses: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
                )
                fig_category.update_layout(
                    legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02},
                    margin={"l": 10, "r": 100, "t": 58, "b": 10},
                )
                st.plotly_chart(fig_category, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No expenses found for the selected categories.")

        st.divider()
        lower_left, lower_right = st.columns([1.1, 1])

        with lower_left:
            st.subheader("Top 5 expenses")
            expense_df = filtered_df[filtered_df["Amount"] < 0].copy()
            if not expense_df.empty:
                expense_df["Amount (₹)"] = -expense_df["Amount"]
                top_expenses = expense_df.nlargest(5, "Amount (₹)").copy()
                top_expenses["Date"] = top_expenses["Date"].dt.strftime("%d %b %Y")
                columns_to_show = ["Date", "Description", "Category", "Amount (₹)"]
                st.dataframe(
                    top_expenses[columns_to_show],
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Amount (₹)": st.column_config.NumberColumn(format="₹%.2f")},
                )
            else:
                st.info("No expense transactions are available.")

        with lower_right:
            st.subheader("Spending insights")
            try:
                recommendations = generate_recommendations(filtered_df)
                if recommendations:
                    for recommendation in recommendations:
                        safe_recommendation = html.escape(str(recommendation))
                        st.markdown(
                            f'<div class="insight-card">{safe_recommendation}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No recommendations available for this data.")
            except Exception as error:
                st.warning(f"Unable to generate spending insights: {error}")
        # Unusual Expense Detection
        st.divider()
        st.subheader("Unusual Expense Detection")

        st.caption(
            "Review expense transactions that differ from the "
            "spending patterns detected by the model."
        )

        try:
            anomaly_df = detect_anomalies(filtered_df)

            flagged_df = anomaly_df[
                anomaly_df["Anomaly"]
            ].copy()

            if flagged_df.empty:
                st.success(
                    "No unusual expense transactions were flagged."
                )

            else:
                st.warning(
                    f"{len(flagged_df)} potentially unusual "
                    "expense transaction(s) flagged. "
                    "Please review these transactions."
                )

                flagged_df = flagged_df.sort_values(
                    "Anomaly_Score",
                    ascending=True
                )

                flagged_df["Date"] = pd.to_datetime(
                    flagged_df["Date"],
                    errors="coerce"
                ).dt.strftime("%Y-%m-%d")

                flagged_df = flagged_df.rename(
                    columns={
                        "Expense_Amount": "Expense (₹)",
                        "Anomaly_Score": "Model Score"
                    }
                )

                st.dataframe(
                    flagged_df[
                        [
                            "Date",
                            "Description",
                            "Category",
                            "Expense (₹)",
                            "Model Score"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        except ValueError as error:
            st.info(str(error))

        except Exception as error:
            st.warning(
                f"Unable to analyse unusual expenses: {error}"
            )

elif page == "Transactions":
    st.subheader("Transaction history")
    st.markdown('<p class="section-note">Search, filter, review, and export the transactions currently loaded in the dashboard.</p>', unsafe_allow_html=True)

    search_col, category_col, type_col = st.columns([1.4, 1, 1])
    with search_col:
        search_text = st.text_input("Search description", placeholder="e.g. Grocery Store")
    with category_col:
        transaction_categories = sorted(filtered_df["Category"].fillna("Unclassified").astype(str).unique().tolist())
        selected_category = st.selectbox("Category", ["All categories"] + transaction_categories)
    with type_col:
        transaction_type = st.selectbox("Transaction type", ["All transactions", "Expenses", "Income"])

    display_df = filtered_df.copy()
    if search_text.strip():
        display_df = display_df[
            display_df["Description"].astype(str).str.contains(search_text.strip(), case=False, na=False, regex=False)
        ]
    if selected_category != "All categories":
        display_df = display_df[display_df["Category"].astype(str) == selected_category]
    if transaction_type == "Expenses":
        display_df = display_df[display_df["Amount"] < 0]
    elif transaction_type == "Income":
        display_df = display_df[display_df["Amount"] > 0]

    st.caption(f"Showing {len(display_df):,} of {len(filtered_df):,} transactions")
    if not display_df.empty:
        table_df = display_df.copy()
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
        table_df = table_df.sort_values("Date", ascending=False)
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
            },
        )
        csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download filtered transactions (CSV)",
            data=csv_data,
            file_name="filtered_transactions.csv",
            mime="text/csv",
            use_container_width=False,
        )
    else:
        st.info("No transactions match these filters. Try changing your search or selections.")


else:  # Category Predictor page
    st.subheader("Category predictor")
    st.markdown('<p class="section-note">Enter a transaction description to estimate its category using the saved machine-learning model.</p>', unsafe_allow_html=True)

    if category_model is None:
        st.warning("The category model was not found. Run `python train_models.py` to train it first.")
    else:
        with st.form("category_prediction_form"):
            description = st.text_input(
                "Transaction description",
                placeholder="Example: DMart Grocery",
            )
            predict_clicked = st.form_submit_button("Predict category", use_container_width=True)

        if predict_clicked:
            if description.strip():
                try:
                    predicted_category = predict_category(category_model, description.strip())
                    st.success(f"Predicted category: {predicted_category}")
                except Exception as error:
                    st.error(f"Could not predict the category: {error}")
            else:
                st.warning("Enter a transaction description first.")


st.divider()
st.caption("Expense Intelligence · Review important financial decisions against your original bank statement.")
