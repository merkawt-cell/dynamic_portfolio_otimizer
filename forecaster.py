import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import zipfile
import tempfile
import h5py
from sklearn.preprocessing import RobustScaler, LabelEncoder
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Region mapping for ETFs
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

# Default region for unknown tickers
DEFAULT_REGION = "North America"

# Default model directory
DEFAULT_MODEL_DIR = "trained_models_LSTM_2000_epochs/trained_models_LSTM_2000_epochs"

# Feature columns used by the model (must match training)
FEATURE_COLS = ['corr_3m', 'max_dd_6m', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'vol_1m', 'Region_Encoded', 'rsi_14', 'position_52w']

# Numeric features for scaling (excludes Region_Encoded which is at index 6)
NUMERIC_FEATURES_INDICES = [0, 1, 2, 3, 4, 5, 7, 8]


def build_lstm_model(input_shape=(10, 9)):
    """Build LSTM model with same architecture as training."""
    from tf_keras.models import Sequential
    from tf_keras.layers import LSTM, Dense, Dropout
    from tf_keras.regularizers import l2

    model = Sequential([
        LSTM(16, input_shape=input_shape, kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(1, kernel_regularizer=l2(0.0001))
    ])
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    return model


def load_model_with_weights(model_path: str):
    """
    Load model by manually extracting weights from .keras file.
    This handles compatibility issues between Keras versions.
    """
    model = build_lstm_model(input_shape=(10, 9))

    with zipfile.ZipFile(model_path, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            weights_path = Path(tmpdir) / 'model.weights.h5'

            with h5py.File(weights_path, 'r') as f:
                # Load LSTM weights
                lstm_kernel = np.array(f['layers/lstm/cell/vars/0'])
                lstm_recurrent = np.array(f['layers/lstm/cell/vars/1'])
                lstm_bias = np.array(f['layers/lstm/cell/vars/2'])

                # Load Dense weights
                dense_kernel = np.array(f['layers/dense/vars/0'])
                dense_bias = np.array(f['layers/dense/vars/1'])

                # Set weights (layers[0]=LSTM, layers[1]=Dropout, layers[2]=Dense)
                model.layers[0].set_weights([lstm_kernel, lstm_recurrent, lstm_bias])
                model.layers[2].set_weights([dense_kernel, dense_bias])

    return model


def fetch_etf_data(tickers: List[str], start_date: str = "2010-01-01") -> pd.DataFrame:
    """Fetch close prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers=tickers, start=start_date, auto_adjust=True, progress=False)

        if data.empty:
            raise ValueError(f"No data returned for tickers: {tickers}")

        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].copy()
            if hasattr(close_prices.columns, 'name'):
                close_prices.columns.name = None
        else:
            if len(tickers) == 1:
                close_prices = data[['Close']].copy()
                close_prices.columns = tickers
            else:
                close_prices = data['Close'].copy()

        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=tickers[0])

        return close_prices.dropna()

    except Exception as e:
        raise ValueError(f"Error fetching data for tickers {tickers}: {str(e)}")


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_features(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from close prices (matches notebook approach)."""
    returns = close_prices.pct_change().dropna()

    # Momentum features
    momentum_1m = close_prices.pct_change(21)
    momentum_3m = close_prices.pct_change(63)
    momentum_6m = close_prices.pct_change(126)

    # Volatility features
    vol_1m = returns.rolling(21).std()

    # 22-day return (target variable for training, feature for context)
    return_22d = close_prices.pct_change(22)

    # Max drawdown
    def max_drawdown(series):
        cumulative = (1 + series).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    drawdown_6m = returns.rolling(126).apply(max_drawdown, raw=False)

    # Correlation features
    corr_3m = returns.rolling(63).corr().groupby(level=0).mean()

    # RSI indicator
    rsi_14 = compute_rsi(close_prices, 14)

    # Position in 52-week range (0 = lowest, 1 = highest)
    high_52w = close_prices.rolling(252).max()
    low_52w = close_prices.rolling(252).min()
    position_52w = (close_prices - low_52w) / (high_52w - low_52w + 1e-8)

    # Combine features
    features = pd.concat({
        "return_22d": return_22d,
        "momentum_1m": momentum_1m,
        "momentum_3m": momentum_3m,
        "momentum_6m": momentum_6m,
        "vol_1m": vol_1m,
        "max_dd_6m": drawdown_6m,
        "corr_3m": corr_3m,
        "rsi_14": rsi_14,
        "position_52w": position_52w
    }, axis=1).dropna()

    return features


def prepare_dataset(features: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Prepare dataset in the format expected by the model."""
    # Flatten column names
    features.columns = [f"{feat}_{ticker}" for feat, ticker in features.columns]

    # Reshape to long format
    dataset = features.reset_index()
    dataset = dataset.melt(id_vars="Date", var_name="Ticker_Feature", value_name="value")
    dataset[["Feature", "Ticker"]] = dataset["Ticker_Feature"].str.rsplit("_", n=1, expand=True)
    dataset = dataset.pivot_table(index=["Date", "Ticker"], columns="Feature", values="value").reset_index()
    dataset = dataset.dropna().reset_index(drop=True)

    # Add region encoding
    le = LabelEncoder()
    le.fit(["Developed Markets", "Emerging Markets", "North America"])
    dataset["Region"] = dataset["Ticker"].map(lambda t: REGION_MAPPING.get(t, DEFAULT_REGION))
    dataset['Region_Encoded'] = le.transform(dataset['Region'])

    return dataset


def load_scalers(model_dir: str) -> Dict[str, RobustScaler]:
    """Load per-ticker scalers from pickle file."""
    scalers_path = Path(model_dir) / "scalers.pkl"
    if scalers_path.exists():
        with open(scalers_path, 'rb') as f:
            return pickle.load(f)
    return {}


def create_sequences_for_prediction(
    data: pd.DataFrame,
    ticker: str,
    scaler: Optional[RobustScaler] = None,
    seq_length: int = 10
) -> Tuple[np.ndarray, Dict]:
    """Create sequences for a single ticker's prediction."""
    ticker_data = data[data['Ticker'] == ticker].sort_values('Date')

    if len(ticker_data) < seq_length:
        return None, None

    features = ticker_data[FEATURE_COLS].values.copy()
    dates = ticker_data['Date'].values
    last_return_22d = ticker_data['return_22d'].values[-1] if 'return_22d' in ticker_data.columns else 0.0

    # Apply scaling if scaler provided
    if scaler is not None:
        features_flat = features.reshape(-1, features.shape[1])
        features_flat[:, NUMERIC_FEATURES_INDICES] = scaler.transform(features_flat[:, NUMERIC_FEATURES_INDICES])
        features = features_flat

    # Get the last sequence for prediction
    X = features[-seq_length:].reshape(1, seq_length, len(FEATURE_COLS))

    info = {
        'ticker': ticker,
        'date': dates[-1],
        'last_return_22d': last_return_22d
    }

    return X, info


def predict_returns(
    tickers: List[str],
    model_path: Optional[str] = None,
    start_date: str = "2010-01-01"
) -> Dict[str, Dict]:
    """
    Predict 22-day returns for given tickers using per-ticker LSTM models.

    Args:
        tickers: List of ETF ticker symbols
        model_path: Path to model directory containing per-ticker models
        start_date: Start date for historical data

    Returns:
        Dictionary with ticker predictions and metadata
    """
    # Determine model directory
    if model_path:
        model_dir = Path(model_path).parent if model_path.endswith('.keras') else Path(model_path)
    else:
        model_dir = Path(DEFAULT_MODEL_DIR)

    # Fetch and prepare data
    close_prices = fetch_etf_data(tickers, start_date)
    features = compute_features(close_prices)
    dataset = prepare_dataset(features, tickers)

    # Sort by ticker and date
    dataset = dataset.sort_values(['Ticker', 'Date'])

    # Load scalers
    scalers = load_scalers(str(model_dir))

    predictions = {}

    # Check if models exist
    models_available = model_dir.exists() and any(model_dir.glob("*_model.keras"))

    if models_available:
        try:
            for ticker in tickers:
                model_file = model_dir / f"{ticker}_model.keras"

                if not model_file.exists():
                    predictions[ticker] = {
                        'predicted_return_22d': None,
                        'last_actual_return_22d': 0.0,
                        'prediction_date': str(dataset[dataset['Ticker'] == ticker]['Date'].max())[:10] if len(dataset[dataset['Ticker'] == ticker]) > 0 else 'N/A',
                        'note': f'Model not found for {ticker}'
                    }
                    continue

                # Get scaler for this ticker
                scaler = scalers.get(ticker)

                # Create sequence for prediction
                X, info = create_sequences_for_prediction(dataset, ticker, scaler, seq_length=10)

                if X is None:
                    predictions[ticker] = {
                        'predicted_return_22d': None,
                        'last_actual_return_22d': 0.0,
                        'prediction_date': 'N/A',
                        'note': f'Insufficient data for {ticker}'
                    }
                    continue

                # Load model using custom loader for Keras version compatibility
                model = load_model_with_weights(str(model_file))
                y_pred = model.predict(X, verbose=0)

                predictions[ticker] = {
                    'predicted_return_22d': float(y_pred[0][0]),
                    'last_actual_return_22d': float(info['last_return_22d']),
                    'prediction_date': str(info['date'])[:10]
                }

        except Exception as e:
            # Fallback if models can't be loaded
            for ticker in tickers:
                ticker_data = dataset[dataset['Ticker'] == ticker]
                if len(ticker_data) > 0:
                    predictions[ticker] = {
                        'predicted_return_22d': None,
                        'last_actual_return_22d': float(ticker_data['return_22d'].values[-1]) if 'return_22d' in ticker_data.columns else 0.0,
                        'prediction_date': str(ticker_data['Date'].values[-1])[:10],
                        'note': f'Error loading model: {str(e)}'
                    }
                else:
                    predictions[ticker] = {
                        'predicted_return_22d': None,
                        'last_actual_return_22d': 0.0,
                        'prediction_date': 'N/A',
                        'note': 'No data available'
                    }
    else:
        # No models available - return historical data only
        for ticker in tickers:
            ticker_data = dataset[dataset['Ticker'] == ticker]
            if len(ticker_data) > 0:
                predictions[ticker] = {
                    'predicted_return_22d': None,
                    'last_actual_return_22d': float(ticker_data['return_22d'].values[-1]) if 'return_22d' in ticker_data.columns else 0.0,
                    'prediction_date': str(ticker_data['Date'].values[-1])[:10],
                    'note': 'Model not loaded - returning historical data only'
                }
            else:
                predictions[ticker] = {
                    'predicted_return_22d': None,
                    'last_actual_return_22d': 0.0,
                    'prediction_date': 'N/A',
                    'note': 'No data available'
                }

    return predictions


def get_expected_returns(tickers: List[str], model_path: Optional[str] = None) -> np.ndarray:
    """
    Get expected returns vector for portfolio optimization.

    Returns annualized expected returns based on 22-day predictions.
    """
    predictions = predict_returns(tickers, model_path)

    mu = []
    for ticker in tickers:
        if ticker in predictions:
            pred = predictions[ticker].get('predicted_return_22d')
            if pred is not None:
                # Annualize 22-day return (approximately 12 periods per year)
                annualized = (1 + pred) ** 12 - 1
                mu.append(annualized)
            else:
                # Fallback to historical return
                hist = predictions[ticker].get('last_actual_return_22d', 0.0)
                annualized = (1 + hist) ** 12 - 1
                mu.append(annualized)
        else:
            mu.append(0.0)

    return np.array(mu)


def compute_covariance_matrix(
    tickers: List[str],
    start_date: str = "2010-01-01",
    annualize: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute historical covariance matrix for given tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        annualize: Whether to annualize the covariance matrix

    Returns:
        Tuple of (covariance matrix as numpy array, returns DataFrame)
    """
    close_prices = fetch_etf_data(tickers, start_date)
    returns = close_prices.pct_change().dropna()

    cov_matrix = returns.cov()

    if annualize:
        # Annualize covariance (252 trading days)
        cov_matrix = cov_matrix * 252

    # Ensure tickers are in correct order
    cov_matrix = cov_matrix.reindex(index=tickers, columns=tickers)

    return cov_matrix.values, returns


def get_portfolio_data(
    tickers: List[str],
    model_path: Optional[str] = None,
    start_date: str = "2010-01-01"
) -> Dict:
    """
    Get all data needed for portfolio optimization.

    Args:
        tickers: List of ETF ticker symbols
        model_path: Path to trained model directory
        start_date: Start date for historical data

    Returns:
        Dictionary with expected returns, covariance matrix, and metadata
    """
    # Get forecasted expected returns
    mu = get_expected_returns(tickers, model_path)

    # Cap extreme returns to reasonable values (max 200% annual)
    mu = np.clip(mu, -0.5, 2.0)

    # Replace NaN with 0
    mu = np.nan_to_num(mu, nan=0.0)

    # Get covariance matrix
    cov, returns = compute_covariance_matrix(tickers, start_date)

    # Handle NaN in covariance matrix
    cov = np.nan_to_num(cov, nan=0.0)

    # Ensure covariance matrix is positive semi-definite
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 0:
        cov = cov + (-min_eig + 1e-6) * np.eye(len(tickers))

    # Get historical statistics
    hist_returns = returns.mean() * 252  # Annualized
    hist_volatility = returns.std() * np.sqrt(252)  # Annualized

    # Replace NaN in historical stats
    hist_returns = hist_returns.fillna(0.0)
    hist_volatility = hist_volatility.fillna(0.0)

    return {
        "tickers": tickers,
        "expected_returns": mu,
        "covariance_matrix": cov,
        "historical_returns": hist_returns.to_dict(),
        "historical_volatility": hist_volatility.to_dict()
    }
