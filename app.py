import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rossmann Prediction", page_icon="🏪", layout="wide")
BASE = Path(__file__).resolve().parent
SALES_MODEL = BASE / "models" / "sales_model.pkl"
CUSTOMER_MODEL = BASE / "models" / "customers_model.pkl"

st.title("🏪 Rossmann Store Sales & Customer Prediction")
st.caption("ML-powered prediction dashboard")

@st.cache_resource
def load_models():
    return joblib.load(SALES_MODEL), joblib.load(CUSTOMER_MODEL)

if not SALES_MODEL.exists() or not CUSTOMER_MODEL.exists():
    st.error("Trained models are missing.")
    st.write("Copy these files into the **models** folder:")
    st.code("models/sales_model.pkl\nmodels/customers_model.pkl")
    st.stop()

sales_model, customer_model = load_models()

def create_features(df):
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a Date column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Invalid Date value found.")

    aliases = {"IsHoliday":"StateHoliday", "IsPromo":"Promo"}
    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]

    defaults = {
        "DayOfWeek": df["Date"].dt.dayofweek + 1, "Promo": 0,
        "StateHoliday": 0, "SchoolHoliday": 0, "StoreType": "a",
        "Assortment": "a", "CompetitionDistance": 0, "Promo2": 0,
        "CompetitionOpenSinceMonth": 1,
        "CompetitionOpenSinceYear": df["Date"].dt.year,
        "Promo2SinceWeek": 1, "Promo2SinceYear": df["Date"].dt.year,
        "PromoInterval": "None"
    }
    for c, v in defaults.items():
        if c not in df.columns: df[c] = v

    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.isocalendar().week.astype(int)
    df["Quarter"] = df.Date.dt.quarter
    df["Weekday"] = df.Date.dt.dayofweek
    df["IsWeekend"] = (df["Weekday"] >= 5).astype(int)
    df["MonthPosition"] = pd.cut(df["Day"], [0,10,20,31],
                                 labels=["Beginning","Mid","End"],
                                 include_lowest=True).astype(str)
    df["IsBeginningOfMonth"] = (df.Day <= 10).astype(int)
    df["IsMidMonth"] = ((df.Day > 10) & (df.Day <= 20)).astype(int)
    df["IsEndOfMonth"] = (df.Day > 20).astype(int)

    df["Day_sin"] = np.sin(2*np.pi*df.Day/31)
    df["Day_cos"] = np.cos(2*np.pi*df.Day/31)
    df["Month_sin"] = np.sin(2*np.pi*df.Month/12)
    df["Month_cos"] = np.cos(2*np.pi*df.Month/12)
    df["Weekday_sin"] = np.sin(2*np.pi*df.Weekday/7)
    df["Weekday_cos"] = np.cos(2*np.pi*df.Weekday/7)

    df["CompetitionDistance"] = pd.to_numeric(df.CompetitionDistance, errors="coerce").fillna(0)
    co_month = pd.to_numeric(df.CompetitionOpenSinceMonth, errors="coerce").fillna(1)
    co_year = pd.to_numeric(df.CompetitionOpenSinceYear, errors="coerce").fillna(df.Year)
    df["CompetitionAge"] = ((df.Year-co_year)*12 + df.Month-co_month).clip(lower=0)
    df["HasCompetition"] = (df.CompetitionDistance > 0).astype(int)
    df["HasPromo2"] = pd.to_numeric(df.Promo2, errors="coerce").fillna(0).astype(int)

    holiday = df.StateHoliday.astype(str).str.lower()
    df["IsStateHoliday"] = (~holiday.isin(["0","0.0","false","nan","none",""])).astype(int)
    df["IsSchoolHoliday"] = pd.to_numeric(df.SchoolHoliday, errors="coerce").fillna(0).astype(int)

    # Safe fallbacks for models trained with lag/rolling columns.
    for c in ["Sales_Lag_1","Sales_Lag_7","Sales_Lag_14","Sales_Lag_28",
              "Sales_Rolling_Mean_7","Sales_Rolling_Mean_14",
              "Sales_Rolling_Mean_28","Sales_Rolling_Std_7"]:
        if c not in df.columns: df[c] = 0
    df["DaysToNextHoliday"] = 0
    df["DaysAfterHoliday"] = 0
    return df

st.sidebar.header("Store Manager Inputs")
store_id = st.sidebar.number_input("Store ID", min_value=1, value=1, step=1)
store_type = st.sidebar.selectbox("Store Type", ["a","b","c","d"])
assortment = st.sidebar.selectbox("Assortment", ["a","b","c"])
competition = st.sidebar.number_input("Competition Distance", min_value=0.0, value=1000.0)
promo2 = st.sidebar.selectbox("Promo2", [0,1])
school = st.sidebar.selectbox("School Holiday", [0,1])
mode = st.sidebar.radio("Input mode", ["CSV Upload","Single Date"])

if mode == "CSV Upload":
    st.subheader("Upload Prediction CSV")
    st.write("Required: Date. Optional: IsHoliday, IsWeekend, IsPromo, SchoolHoliday and model inputs.")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f is None:
        st.info("Upload a CSV to generate predictions.")
        st.stop()
    raw = pd.read_csv(f)
else:
    d = st.date_input("Prediction Date")
    raw = pd.DataFrame([{
        "Store":store_id, "Date":pd.Timestamp(d), "StoreType":store_type,
        "Assortment":assortment, "CompetitionDistance":competition,
        "Promo2":promo2, "Promo":0, "StateHoliday":0, "SchoolHoliday":school
    }])

if "Store" not in raw.columns: raw["Store"] = store_id
if "StoreType" not in raw.columns: raw["StoreType"] = store_type
if "Assortment" not in raw.columns: raw["Assortment"] = assortment
if "CompetitionDistance" not in raw.columns: raw["CompetitionDistance"] = competition
if "Promo2" not in raw.columns: raw["Promo2"] = promo2
if "SchoolHoliday" not in raw.columns: raw["SchoolHoliday"] = school

if st.button("🚀 Generate Predictions", type="primary"):
    try:
        x = create_features(raw)
        x = x.drop(columns=[c for c in ["Sales","Customers"] if c in x.columns])
        sales = np.maximum(0, sales_model.predict(x))
        customers = np.maximum(0, customer_model.predict(x))
        result = raw.copy()
        result["Predicted_Sales"] = sales
        result["Predicted_Customers"] = customers

        a,b,c = st.columns(3)
        a.metric("Total Predicted Sales", f"₹{sales.sum():,.0f}")
        b.metric("Total Predicted Customers", f"{customers.sum():,.0f}")
        c.metric("Average Daily Sales", f"₹{sales.mean():,.0f}")

        st.subheader("Prediction Results")
        st.dataframe(result, use_container_width=True)

        plot = result.copy()
        plot["Date"] = pd.to_datetime(plot["Date"])
        plot = plot.sort_values("Date")

        st.subheader("📈 Predicted Sales")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(plot["Date"], plot["Predicted_Sales"], marker="o")
        ax.set_xlabel("Date"); ax.set_ylabel("Predicted Sales"); ax.grid(True, alpha=.3)
        st.pyplot(fig)

        st.subheader("👥 Predicted Customers")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(plot["Date"], plot["Predicted_Customers"], marker="o")
        ax.set_xlabel("Date"); ax.set_ylabel("Predicted Customers"); ax.grid(True, alpha=.3)
        st.pyplot(fig)

        st.subheader("📊 Sales & Customers Trend")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(plot["Date"], plot["Predicted_Sales"], marker="o", label="Sales")
        ax2 = ax.twinx()
        ax2.plot(plot["Date"], plot["Predicted_Customers"], marker="s", label="Customers")
        ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax2.set_ylabel("Customers")
        st.pyplot(fig)

        st.download_button("⬇️ Download Predictions as CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           "rossmann_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure the saved pipelines use the same feature names and preprocessing as training.")

st.markdown("---")
st.caption("Rossmann Store Sales Prediction | Streamlit Community Cloud")
