import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")


def download_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """Download historical stock data for a ticker."""
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker} (period={period}).")
    return data


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build lag features and target values for regression."""
    data = df[['Close']].copy()
    for lag in range(1, window + 1):
        data[f'lag_{lag}'] = data['Close'].shift(lag)

    data['ma_5'] = data['Close'].rolling(window=5).mean()
    data['ma_10'] = data['Close'].rolling(window=10).mean()
    data['daily_return'] = data['Close'].pct_change()
    data['target_next'] = data['Close'].shift(-1)

    data = data.dropna()
    return data


def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': model.score(X, y)
    }


if __name__ == '__main__':
    ticker = 'RELIANCE.NS'  # change to any symbol like 'TCS.NS', 'HDFCBANK.NS', 'AAPL', 'MSFT'
    print(f'Downloading {ticker} prices...')

    df = download_data(ticker, period='5y')
    features = build_features(df, window=5)

    X = features.drop(columns=['target_next'])
    y = features['target_next']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = train_model(X_train, y_train)

    train_metrics = evaluate(model, X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)

    print("Train metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    y_test_pred = model.predict(X_test)

    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test, label='Actual Close')
    plt.plot(y_test.index, y_test_pred, label='Predicted Close', alpha=0.8)
    plt.title(f'{ticker} Stock Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # predict next day from last row
    last_row = X.iloc[-1:].copy()
    next_price_estimate = model.predict(last_row)[0]
    print(f'Next-day estimated closing price: {next_price_estimate:.2f}')
