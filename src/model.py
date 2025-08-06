import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class PriceModel:
    def __init__(self, df: pd.DataFrame, target_col: str = "Target"):
        self.df = df.copy()
        self.target_col = target_col
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_test_split(self, test_size=0.2):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Time series split: no shuffle!
        split_index = int(len(self.df) * (1 - test_size))
        self.X_train, self.X_test = X.iloc[:split_index], X.iloc[split_index:]
        self.y_train, self.y_test = y.iloc[:split_index], y.iloc[split_index:]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f"✅ Evaluation: MSE = {mse:.4f}, R² = {r2:.4f}")
        return predictions

    def save_model(self, path="outputs/price_model.pkl"):
        joblib.dump(self.model, path)
