# Stock Price Prediction (Streamlit + LSTM + Linear Regression)

This repository contains a beginner-friendly stock price prediction project with two modes:
- `stock_price_prediction_project.py`: standalone script for local use
- `stock_price_prediction_streamlit.py`: interactive Streamlit app for deployment

## Features
- Historical data from Yahoo Finance (`yfinance`)
- Data preprocessing (missing value fill, scaling)
- Linear Regression and LSTM model training
- Evaluation with RMSE
- Actual vs predicted plots
- Next-day prediction

## Run locally
```bash
cd "C:\Users\hp\OneDrive\文档\NSE 2"
pip install -r requirements.txt
python stock_price_prediction_project.py
```

## Run Streamlit
```bash
cd "C:\Users\hp\OneDrive\文档\NSE 2"
python -m streamlit run stock_price_prediction_streamlit.py
```

## Deploy to GitHub Pages with Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. In Streamlit Cloud, create a new app from the repo.
3. Specify `stock_price_prediction_streamlit.py` as the entrypoint.

## Project files
- `stock_price_prediction.py` (initial linear model script)
- `stock_price_prediction_project.py` (full workflow)
- `stock_price_prediction_streamlit.py` (Streamlit app)
- `requirements.txt`
- `.gitignore`
- `README.md`
