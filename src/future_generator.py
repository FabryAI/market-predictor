import pandas as pd

class FeatureGenerator:
    def __init__(self, df: pd.DataFrame, target_col: str = "Target", lags: int = 5):
        self.df = df.copy()
        self.target_col = target_col
        self.lags = lags

    def add_lag_features(self):
        """
        Aggiunge colonne con i valori precedenti (lag).
        """
        for i in range(1, self.lags + 1):
            self.df[f'Close_lag_{i}'] = self.df['Close'].shift(i)
            self.df[f'Volume_lag_{i}'] = self.df['Volume'].shift(i)
            self.df[f'Return_lag_{i}'] = self.df['Close'].pct_change().shift(i)
        return self

    def get_dataset(self):
        """
        Restituisce dataset pronto per ML (senza NaN).
        """
        return self.df.dropna()
