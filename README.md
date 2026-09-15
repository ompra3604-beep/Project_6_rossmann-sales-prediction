# Rossmann Streamlit Deployment

## Put your trained models here

Copy your trained complete scikit-learn pipelines to:

```text
models/sales_model.pkl
models/customers_model.pkl
```

Example:

```python
import joblib
joblib.dump(sales_pipeline, "models/sales_model.pkl")
joblib.dump(customer_pipeline, "models/customers_model.pkl")
```

The saved object should include preprocessing (imputation/encoding/scaling) and the regression model.

## Project structure

```text
rossmann_streamlit_deployment/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── sales_model.pkl
│   └── customers_model.pkl
└── data/
```

## Test locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the two `.pkl` files.
3. Open Streamlit Community Cloud.
4. Connect GitHub.
5. Select your repository and `app.py`.
6. Click Deploy.
7. Submit the generated `https://....streamlit.app` link.

## CSV

Minimum:

```csv
Date
2026-09-15
2026-09-16
2026-09-17
```

Recommended:

```csv
Date,IsHoliday,IsWeekend,IsPromo,SchoolHoliday
2026-09-15,0,0,1,0
2026-09-16,0,0,1,0
2026-09-17,0,0,0,0
```

## Assignment requirements covered

- Store ID input
- CSV upload
- Date-dependent inputs
- Other model inputs
- Sales prediction
- Customer prediction
- Sales and customer plots
- Combined trend plot
- CSV download

### Important

If your model was trained with lag/rolling sales features, the app currently uses safe fallback values when those columns are absent. For a production model, calculate lag/rolling values from historical store data during inference.
