import pandas as pd

class PricePreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean(self):
        """
        Rimuove valori nulli o righe con volume 0.
        """
        self.df = self.df.dropna()
        self.df = self.df[self.df['Volume'] > 0]
        return self

    def add_features(self):
        """
        Aggiunge feature tecniche di base.
        """
        self.df['SMA_10'] = self.df['Close'].rolling(window=10).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['Return_1h'] = self.df['Close'].pct_change(periods=60)
        self.df['Volatility'] = self.df['Close'].rolling(window=60).std()
        return self

    def add_target(self, shift_periods=60):
        """
        Crea colonna target: prezzo close futuro (es: 1 ora dopo).
        """
        self.df['Target'] = self.df['Close'].shift(-shift_periods)
        return self

    def get_processed(self):
        return self.df.dropna()
    
    def add_advanced_features(self):
        df = self.df.copy()

        # Bande di Bollinger
        rolling_mean = df["Close"].rolling(window=20)
        df["BB_Middle"] = rolling_mean.mean()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(window=20).std()
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        df["TR"] = df[["High", "Close"]].max(axis=1) - df[["Low", "Close"]].min(axis=1)
        df["ATR_14"] = df["TR"].rolling(window=14).mean()

        # Medie mobili
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Momentum semplice
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

        # Z-Score
        df["Z_score_30"] = (
            (df["Close"] - df["Close"].rolling(window=30).mean()) /
            df["Close"].rolling(window=30).std()
        )

        # Range ultimi 60 giorni
        df["Rolling_max_60"] = df["Close"].rolling(window=60).max()
        df["Rolling_min_60"] = df["Close"].rolling(window=60).min()
        df["Range_60"] = df["Rolling_max_60"] - df["Rolling_min_60"]

        # Body size e bullish candle
        df["Body_size"] = df["Close"] - df["Open"]
        df["Is_bullish"] = (df["Close"] > df["Open"]).astype(int)

        # Variazione volume
        df["Volume_change"] = df["Volume"].pct_change().fillna(0)

        self.df = df.dropna()
        return self


