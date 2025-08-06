from src.data_loader import PriceDataLoader
from src.future_generator import FeatureGenerator
from src.preprocessing import PricePreprocessor
from src.price_model_xgb import PriceModelXGB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # 📥 Load raw data
    loader = PriceDataLoader("data/raw/btcusd_1-min_data.csv")
    df = loader.load_data()

    # ✅ Aggregate to daily candles
    df_daily = df.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    print(f"✅ Aggregated to daily candles: {len(df_daily)} rows")
    print(df_daily.head())

    # 💾 Save cleaned and aggregated data
    loader.df = df_daily  # reuse the object to save
    loader.save_processed("data/processed/btcusd_clean_daily.csv")

    # 🧹 Preprocessing + Base Feature Engineering + Target
    pre = PricePreprocessor(df_daily)
    df_processed = (
        pre.clean()
        .add_features()
        .add_advanced_features()  # 👈 add custom indicators here
        .add_target(shift_periods=1)  # predict 1 day ahead
        .get_processed()
    )

    print("✅ Base features and target created.")
    print(df_processed.head())
    df_processed.to_csv("data/processed/btcusd_features_daily.csv")

    # 🧠 Generate lag features
    fg = FeatureGenerator(df_processed, lags=5)
    df_final = fg.add_lag_features().get_dataset()

    print("✅ Final dataset ready for training:")
    print(df_final.head())
    df_final.to_csv("data/processed/btcusd_final_dataset_daily.csv")

    # 📈 Train model with XGBoost
    model = PriceModelXGB(df_final, target_col="Target")
    model.train_test_split(test_size=0.2)
    model.train()
    model.evaluate()
    model.save_model()

    # ✅ Get predictions
    preds = model.model.predict(model.X_test)

    # 📏 Calculate evaluation metrics
    mse = mean_squared_error(model.y_test, preds)
    r2 = r2_score(model.y_test, preds)
    mae = mean_absolute_error(model.y_test, preds)

    print(f"📏 MSE: {mse:.4f}")
    print(f"📏 MAE: {mae:.4f}")
    print(f"📊 R²: {r2:.4f}")

    # 📊 Plot predictions vs actual values
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(model.y_test.values[:200], label="Actual values", linewidth=2)
    plt.plot(preds[:200], label="Predictions", linestyle='--')
    plt.legend()
    plt.title("Predictions vs Actual values (first 200 days)")
    plt.xlabel("Time (days)")
    plt.ylabel("BTC Closing Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/predictions_vs_real.png")
    plt.show()

    print("✅ Chart saved to outputs/predictions_vs_real.png")
