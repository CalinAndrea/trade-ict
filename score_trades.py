import pandas as pd
import joblib

# === Load trained model ===
model = joblib.load("trade_model.pkl")

# === Load new trade data ===
df = pd.read_csv("weekly_trades_ml_dataset.csv")  # Replace with new week's CSV

# === Feature preparation ===
features = ['hour', 'direction_code', 'duration', 'fvg_size']
X = df[features]

# === Predict probabilities ===
df['win_prob'] = model.predict_proba(X)[:, 1]

# === Filter trades by confidence threshold ===
confident_trades = df[df['win_prob'] > 0.6].copy()

# === Save or view top trades ===
confident_trades = confident_trades.sort_values(by='win_prob', ascending=False)
print(confident_trades[['timestamp', 'direction', 'entry', 'win_prob', 'target']].head(10))

confident_trades.to_csv("filtered_high_confidence_trades.csv", index=False)
print("✅ Saved filtered trades to filtered_high_confidence_trades.csv")
