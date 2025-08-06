import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

class PriceModelXGB:
    def __init__(self, df: pd.DataFrame, target_col: str = "Target"):
        self.df = df.dropna()
        self.target_col = target_col
        self.model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def train_test_split(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, shuffle=False
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        print(f"✅ XGBoost Evaluation: MSE = {mse:.4f}, R² = {r2:.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename="outputs/xgb_model.pkl"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(self.model, filename)
        print(f"✅ Modello XGBoost salvato in {filename}")

    def load_model(self, filename="outputs/xgb_model.pkl"):
        self.model = joblib.load(filename)
