import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import streamlit as st

try:
    import yfinance as yf
except ImportError:
    raise ImportError('Please install yfinance: pip install yfinance')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ImportError:
    tf = None


@st.cache_data
def download_stock_data(symbol='AAPL', period='5y'):
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise ValueError('No data downloaded. Check symbol and internet access.')
    return df


def preprocess_data(df):
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    data = df[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close_scaled'] = scaler.fit_transform(data)
    return data, scaler


def create_lstm_sequences(data, sequence_len=60):
    x, y = [], []
    scaled = data['Close_scaled'].values
    for i in range(sequence_len, len(scaled)):
        x.append(scaled[i-sequence_len:i])
        y.append(scaled[i])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def train_linear_regression(data):
    df = data.copy()
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
    return model, X_test, y_test, y_pred, rmse


def train_lstm(data, sequence_len=60, epochs=20, batch_size=32):
    if tf is None:
        raise RuntimeError('TensorFlow is required for LSTM model')
    x, y = create_lstm_sequences(data, sequence_len=sequence_len)
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
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, x_test, y_test, y_pred, rmse, split


def plot_time_series(df, x, y_actual, y_pred, title):
    plt.figure(figsize=(12, 4))
    plt.plot(df.index[-len(y_actual):], y_actual, label='Actual')
    plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def main():
    st.title('Stock Price Prediction (Linear Regression + LSTM)')
    st.write('A Streamlit app to train and compare stock forecasting models.')

    symbol = st.sidebar.text_input('Ticker symbol', 'AAPL')
    period = st.sidebar.selectbox('History period', ['1y', '2y', '5y', '10y'], index=2)
    model_type = st.sidebar.selectbox('Model', ['Linear Regression', 'LSTM'])
    if model_type == 'LSTM':
        lstm_seq = st.sidebar.slider('LSTM sequence length', 30, 120, 60)
        lstm_epochs = st.sidebar.slider('LSTM epochs', 5, 50, 15)
        batch_size = st.sidebar.selectbox('Batch size', [16, 32, 64], index=1)

    if st.sidebar.button('Run prediction'):
        with st.spinner('Downloading and processing data...'):
            df = download_stock_data(symbol=symbol, period=period)
            data, scaler = preprocess_data(df)

        st.subheader('Sample historical data')
        st.dataframe(df.tail(10))

        st.subheader('Close price trend')
        st.line_chart(df['Close'])

        if model_type == 'Linear Regression':
            model, X_test, y_test, y_pred, rmse = train_linear_regression(data)
            st.write(f'Linear Regression RMSE: {rmse:.4f}')
            plot_time_series(data, X_test, y_test, y_pred, f'{symbol} Linear Regression Forecast')

            forecast_input = data[['Close']].values[-5:].reshape(1, -1)
            next_price = model.predict(forecast_input)[0]
            st.success(f'Next-day predicted close price (LinearRegression): {next_price:.2f}')

        else:
            model, x_test, y_test, y_pred, rmse, split = train_lstm(data, sequence_len=lstm_seq, epochs=lstm_epochs, batch_size=batch_size)
            st.write(f'LSTM RMSE: {rmse:.4f}')
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            test_dates = data.index[lstm_seq + split:]
            plt.figure(figsize=(12, 4))
            plt.plot(test_dates, y_test_inv, label='Actual')
            plt.plot(test_dates, y_pred_inv, label='Predicted')
            plt.title(f'{symbol} LSTM Forecast')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            most_recent = data['Close_scaled'].values[-lstm_seq:].reshape(1, lstm_seq, 1)
            next_price_scaled = model.predict(most_recent)
            next_price = scaler.inverse_transform(next_price_scaled)[0][0]
            st.success(f'Next-day predicted close price (LSTM): {next_price:.2f}')

        st.info('Deployment: run with `streamlit run stock_price_prediction_streamlit.py`')


if __name__ == '__main__':
    main()