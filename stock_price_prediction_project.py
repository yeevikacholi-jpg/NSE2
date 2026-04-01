import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# TensorFlow / Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

try:
    import yfinance as yf
except ImportError:
    raise ImportError('Please install yfinance (pip install yfinance)')


def download_stock_data(symbol='AAPL', period='5y'):
    """Download historical stock prices from Yahoo Finance."""
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise ValueError('No data downloaded. Check symbol and internet access.')
    return df


def preprocess_data(df):
    """Handle missing values, select features, and normalize close price."""
    df = df.copy()
    # fill missing values via forward/backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # We only use Close price for simplicity (can expand to more features)
    data = df[['Close']]

    # Scale values to range [0,1] for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close_scaled'] = scaler.fit_transform(data)

    return data, scaler


def create_lstm_sequences(data, sequence_len=60):
    """Convert time series into input-output sequences for LSTM."""
    x = []
    y = []
    scaled = data['Close_scaled'].values
    for i in range(sequence_len, len(scaled)):
        x.append(scaled[i-sequence_len:i])
        y.append(scaled[i])

    x = np.array(x)
    y = np.array(y)

    # LSTM expects 3D input: [samples, timesteps, features]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def train_linear_regression(data):
    """Train a simple linear regression model using lagged features."""
    df = data.copy()
    # Create lag features (past 5 days)
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)

    features = [f'lag_{lag}' for lag in range(1, 6)]
    X = df[features].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'rmse': rmse,
        'features': features,
    }


def train_lstm(data, sequence_len=60, epochs=20, batch_size=32):
    """Train an LSTM model on scaled close prices."""
    x, y = create_lstm_sequences(data, sequence_len=sequence_len)

    # Split with 80% training, 20% testing
    split = int(x.shape[0] * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        'model': model,
        'x_test': x_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'rmse': rmse,
        'split': split,
    }


def plot_results_linear(df, lr_result):
    """Plot actual vs predicted prices for Linear Regression."""
    # Build a date index for test set
    df_lr = df.copy()
    df_lr = df_lr.iloc[-len(lr_result['y_test']):]

    plt.figure(figsize=(12, 6))
    plt.plot(df_lr.index, lr_result['y_test'], label='Actual Close')
    plt.plot(df_lr.index, lr_result['y_pred'], label='Predicted Close')
    plt.title('Linear Regression: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_results_lstm(data, lstm_result, scaler, sequence_len=60):
    """Plot actual vs predicted close prices for LSTM (inverse-scaled)."""
    # Build full range of data for plotting
    x_full = lstm_result['x_test']
    # flatten predictions and reverse scaler
    y_pred = scaler.inverse_transform(lstm_result['y_pred'])
    y_test = scaler.inverse_transform(lstm_result['y_test'].reshape(-1, 1))

    test_dates = data.index[sequence_len + lstm_result['split']:]  # corresponding dates

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual Close')
    plt.plot(test_dates, y_pred, label='Predicted Close')
    plt.title('LSTM: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_next_day_lstm(model, data, scaler, sequence_len=60):
    """Predict next-day close price with trained LSTM model."""
    recent_data = data['Close_scaled'][-sequence_len:].values
    recent_data = recent_data.reshape(1, sequence_len, 1)
    pred = model.predict(recent_data)
    next_price = scaler.inverse_transform(pred)[0][0]
    return next_price


if __name__ == '__main__':
    symbol = 'AAPL'  # change symbol as needed, e.g. 'RELIANCE.NS'

    # Download and preprocess data
    df = download_stock_data(symbol=symbol, period='5y')
    data, scaler = preprocess_data(df)

    # Visualize raw closing price trend
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f'{symbol} Closing Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 1) Linear Regression path
    lr_result = train_linear_regression(data)
    print(f"Linear Regression RMSE: {lr_result['rmse']:.4f}")
    plot_results_linear(data, lr_result)

    # 2) LSTM path
    lstm_result = train_lstm(data, sequence_len=60, epochs=15, batch_size=32)
    print(f"LSTM RMSE: {lstm_result['rmse']:.4f}")
    plot_results_lstm(data, lstm_result, scaler, sequence_len=60)

    # Predict next-day price with LSTM
    next_price = predict_next_day_lstm(lstm_result['model'], data, scaler, sequence_len=60)
    print(f"Next day predicted close price (LSTM): {next_price:.2f}")
