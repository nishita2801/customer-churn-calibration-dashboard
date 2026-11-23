# Customer Churn Probability & Calibration Dashboard

A Streamlit learning project that compares churn probability scores from Logistic Regression and Random Forest, before and after Platt and isotonic calibration.

## Features

- Model comparison on a held-out test set
- Platt and isotonic calibration
- Reliability diagram
- ECE, Brier score, log loss, and ROC AUC
- Example churn probability scores
- Brief findings displayed in the dashboard

## Dataset

This project uses the IBM Telco Customer Churn sample dataset. The CSV is expected at:

`data/raw/Telco-Customer-Churn.csv`

Dataset source: [IBM Telco Customer Churn project](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Run locally on Windows

From the project folder, open PowerShell and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py

Project structure

CustomerChurnCalibration/
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
└── src/
    └── churn_calibration/
        ├── __init__.py
        ├── data.py
        ├── metrics.py
        └── modeling.py

     Evaluation notes
The app uses a stratified train/test split. Preprocessing and calibration are fitted using training data; the held-out test set is used for evaluation. Lower ECE, Brier score, and log loss indicate better probability quality; higher ROC AUC indicates stronger ranking.

This is an educational project using a sample dataset. Results describe this dataset and should not be treated as guarantees about other customers.


