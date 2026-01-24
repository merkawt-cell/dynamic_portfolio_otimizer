"""
ETF Forecasting Model Training Script

This script trains an LSTM model to predict 22-day ETF returns
and saves it for use with the forecasting API.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
from tf_keras.callbacks import EarlyStopping


# Configuration
TICKERS = ["PSI", "IYW", "RING", "PICK", "NLR", "UTES", "LIT", "NANR", "GUNR", "XCEM", "PTLC", "FXU"]
START_DATE = "2015-01-01"
SEQ_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 16

# Region mapping
REGION_MAPPING = {
    "PSI": "North America",
    "IYW": "North America",
    "RING": "Developed Markets",
    "PICK": "Developed Markets",
    "NLR": "Developed Markets",
    "UTES": "North America",
    "LIT": "Developed Markets",
    "NANR": "North America",
    "GUNR": "Developed Markets",
    "XCEM": "Emerging Markets",
    "PTLC": "North America",
    "FXU": "North America",
}


def download_data(tickers, start_date):
    """Download ETF data from Yahoo Finance."""
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers=tickers, start=start_date, auto_adjust=True, progress=False)
    close_prices = data["Close"].dropna()
    print(f"Downloaded {len(close_prices)} rows of data")
    return close_prices


def compute_features(close_prices):
    """Compute technical features."""
    print("Computing features...")
    returns = close_prices.pct_change().dropna()

    momentum_1m = close_prices.pct_change(21)
    momentum_3m = close_prices.pct_change(63)
    momentum_6m = close_prices.pct_change(126)

    vol_1m = returns.rolling(21).std()
    return_22d = close_prices.pct_change(22)

    def max_drawdown(series):
        cumulative = (1 + series).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    drawdown_6m = returns.rolling(126).apply(max_drawdown, raw=False)
    corr_3m = returns.rolling(63).corr().groupby(level=0).mean()

    features = pd.concat({
        "return_22d": return_22d,
        "momentum_1m": momentum_1m,
        "momentum_3m": momentum_3m,
        "momentum_6m": momentum_6m,
        "vol_1m": vol_1m,
        "max_dd_6m": drawdown_6m,
        "corr_3m": corr_3m,
    }, axis=1).dropna()

    return features


def prepare_dataset(features, tickers):
    """Prepare dataset for model training."""
    print("Preparing dataset...")
    features.columns = [f"{feat}_{ticker}" for feat, ticker in features.columns]

    dataset = features.reset_index()
    dataset = dataset.melt(id_vars="Date", var_name="Ticker_Feature", value_name="value")
    dataset[["Feature", "Ticker"]] = dataset["Ticker_Feature"].str.rsplit("_", n=1, expand=True)
    dataset = dataset.pivot_table(index=["Date", "Ticker"], columns="Feature", values="value").reset_index()
    dataset = dataset.dropna().reset_index(drop=True)

    dataset["Region"] = dataset["Ticker"].map(lambda t: REGION_MAPPING.get(t, "North America"))

    return dataset


def scale_and_encode(dataset):
    """Scale numeric features and encode categorical ones."""
    print("Scaling features...")
    df = dataset.copy()

    numeric_features = ['corr_3m', 'max_dd_6m', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'vol_1m']

    scaler = RobustScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    le = LabelEncoder()
    le.fit(["Developed Markets", "Emerging Markets", "North America"])
    df['Region_Encoded'] = le.transform(df['Region'])

    return df, scaler, le


def create_sequences(data, seq_length=10):
    """Create sequences for LSTM training."""
    X, y = [], []
    info = []

    feature_cols = ['corr_3m', 'max_dd_6m', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'vol_1m', 'Region_Encoded']

    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
        features = ticker_data[feature_cols].values
        target = ticker_data['return_22d'].values
        dates = ticker_data['Date'].values

        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])
            info.append({'ticker': ticker, 'date': dates[i+seq_length]})

    return np.array(X), np.array(y), pd.DataFrame(info)


def split_data(df_model, seq_length=10):
    """Split data into train, validation, and test sets."""
    print("Splitting data...")
    X_train_final, X_val, X_test = [], [], []
    y_train_final, y_val, y_test = [], [], []

    for ticker in df_model['Ticker'].unique():
        ticker_data = df_model[df_model['Ticker'] == ticker].sort_values('Date')

        ticker_X, ticker_y, _ = create_sequences(
            ticker_data.reset_index(drop=True),
            seq_length=seq_length
        )

        split_idx = int(len(ticker_X) * 0.8)
        train_X, test_X_ticker = ticker_X[:split_idx], ticker_X[split_idx:]
        train_y, test_y_ticker = ticker_y[:split_idx], ticker_y[split_idx:]

        val_split_idx = int(len(train_X) * 0.8)
        X_train_final.append(train_X[:val_split_idx])
        X_val.append(train_X[val_split_idx:])
        y_train_final.append(train_y[:val_split_idx])
        y_val.append(train_y[val_split_idx:])

        X_test.append(test_X_ticker)
        y_test.append(test_y_ticker)

    X_train_final = np.concatenate(X_train_final)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)
    y_train_final = np.concatenate(y_train_final)
    y_val = np.concatenate(y_val)
    y_test = np.concatenate(y_test)

    print(f"Train: {X_train_final.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return X_train_final, X_val, X_test, y_train_final, y_val, y_test


def build_lstm_model(seq_length, n_features):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """Train the model."""
    print(f"Training model for up to {epochs} epochs...")

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop]
    )

    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    return test_loss, test_mae


def main():
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Download and prepare data
    close_prices = download_data(TICKERS, START_DATE)
    features = compute_features(close_prices)
    dataset = prepare_dataset(features, TICKERS)
    scaled_data, scaler, le = scale_and_encode(dataset)

    # Prepare for training
    df_model = scaled_data.sort_values(['Ticker', 'Date'])

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_model, SEQ_LENGTH)

    # Build and train model
    n_features = X_train.shape[2]
    model = build_lstm_model(SEQ_LENGTH, n_features)
    model.summary()

    train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    model_path = models_dir / "etf_lstm_model.keras"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    print("\nTraining complete! You can now use the model with the /forecast API endpoint.")
    print(f'Example: {{"tickers": ["IYW", "PSI"], "model_path": "{model_path}"}}')


if __name__ == "__main__":
    main()
