"""Task 3: CSV-based sales forecasting interface."""
from pathlib import Path
import sys, joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import merge_and_clean
from src.features import make_features
MODEL_PATH=ROOT/"models"/"latest_sales_model.pkl"; STORE_PATH=ROOT/"data"/"raw"/"store.csv"
st.set_page_config(page_title="Rossmann Sales Forecast",layout="wide");st.title("Rossmann store sales forecast");st.caption("Upload future store-day records to obtain sales forecasts and a 90% empirical interval.")
if not MODEL_PATH.exists(): st.error("Train a model first: python -m src.train");st.stop()
if not STORE_PATH.exists(): st.error("Missing data/raw/store.csv");st.stop()
file=st.file_uploader("Forecast input CSV",type="csv")
if file:
    raw=pd.read_csv(file); required={"Store","Date","Open","Promo","StateHoliday","SchoolHoliday"}; missing=required-set(raw.columns)
    if missing: st.error(f"Missing columns: {', '.join(sorted(missing))}");st.stop()
    features=make_features(merge_and_clean(raw,pd.read_csv(STORE_PATH))); bundle=joblib.load(MODEL_PATH); pred=bundle["pipeline"].predict(features);pred[features.Open.to_numpy()==0]=0; interval=bundle["uncertainty_90"]
    customer_pred=bundle["customer_pipeline"].predict(features); customer_pred[features.Open.to_numpy()==0]=0
    result=raw.copy();result["PredictedSales"]=pred.round(2);result["PredictedCustomers"]=customer_pred.round().astype(int);result["Lower90"]=(pred-interval).clip(0).round(2);result["Upper90"]=(pred+interval).round(2)
    st.line_chart(result.assign(Date=pd.to_datetime(result.Date)).set_index("Date")[["PredictedSales","Lower90","Upper90"]]);st.dataframe(result,use_container_width=True);st.download_button("Download predictions CSV",result.to_csv(index=False).encode(),"sales_predictions.csv","text/csv")
else: st.info("Required fields: Store, Date, Open, Promo, StateHoliday, SchoolHoliday.")
