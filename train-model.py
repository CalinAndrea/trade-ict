import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Load and prepare data ===
df = pd.read_csv("weekly_trades_ml_dataset.csv")
features = ['hour', 'direction_code', 'duration', 'fvg_size']
X = df[features]
y = df['target']

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = clf.predict(X_test)

print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model (optional)
import joblib
joblib.dump(clf, "trade_model.pkl")
print("\n💾 Saved model to trade_model.pkl")
