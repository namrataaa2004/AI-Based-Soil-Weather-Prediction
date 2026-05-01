import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

plt.style.use("seaborn-v0_8")

# --- Weather: CSV columns match your dataset ---
# record_id, date, city, temperature_c, humidity_pct, pressure_hpa,
# wind_speed_kmh, rainfall_mm, cloud_cover_pct, weather
weather_df = pd.read_csv("weather_data.csv")

weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df = weather_df.sort_values("date").reset_index(drop=True)

# One temperature per row (your sample has one city per date)
temp_series = weather_df["temperature_c"].astype(float).values

split_index = int(len(temp_series) * 0.8)
train_temp = temp_series[:split_index]
test_temp = temp_series[split_index:]

arima_order = (5, 1, 0)
arima_model = ARIMA(train_temp, order=arima_order)
arima_result = arima_model.fit()

arima_forecast = arima_result.forecast(steps=len(test_temp))

arima_rmse = np.sqrt(mean_squared_error(test_temp, arima_forecast))
arima_r2 = r2_score(test_temp, arima_forecast)
print("=== ARIMA Results ===")
print(f"RMSE: {arima_rmse:.3f}")
print(f"R2  : {arima_r2:.3f}")

plt.figure(figsize=(10, 4))
plt.plot(weather_df["date"][split_index:], test_temp, label="Actual Temp")
plt.plot(weather_df["date"][split_index:], arima_forecast, label="ARIMA Predicted Temp")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("ARIMA Temperature Forecast")
plt.legend()
plt.tight_layout()
plt.show()


def create_sequences(values, window_size=7):
    """
    Build time-series sequences.

    values: 1D array (T,) or 2D array (T, n_features)
    Returns:
      X: (n_samples, window_size, n_features)
      y: (n_samples,) for 1D input, or (n_samples, n_targets) for 2D targets
    """
    values = np.asarray(values)
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    return np.asarray(X), np.asarray(y)


tf.random.set_seed(42)
np.random.seed(42)

# Proper time-series LSTM (multivariate):
# - split on time (already done with split_index)
# - fit scaler ONLY on training rows (avoid leakage)
# - use multiple weather features to predict temperature
window_size = 14
feature_cols_lstm = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "rainfall_mm",
    "cloud_cover_pct",
]

missing = [c for c in feature_cols_lstm if c not in weather_df.columns]
if missing:
    raise ValueError(f"Missing columns for LSTM: {missing}")

features = weather_df[feature_cols_lstm].astype(float).values  # (T, n_features)

scaler = MinMaxScaler()
scaler.fit(features[:split_index])
features_scaled = scaler.transform(features)

X_all, y_all = create_sequences(features_scaled, window_size)  # y_all is next-step features

# Predict next-step temperature (scaled) as a single output
y_all_temp = y_all[:, 0]

# Each sequence target y_all_temp[i] corresponds to original time index (i + window_size)
target_time_index = np.arange(window_size, len(features))
train_mask = target_time_index < split_index
test_mask = target_time_index >= split_index

X_train = X_all[train_mask]
y_train = y_all_temp[train_mask]
X_test = X_all[test_mask]
y_test = y_all_temp[test_mask]

# Time-based validation split (last 10% of training sequences)
val_size = max(1, int(len(X_train) * 0.1))
X_train_lstm, X_val_lstm = X_train[:-val_size], X_train[-val_size:]
y_train_lstm, y_val_lstm = y_train[:-val_size], y_train[-val_size:]
X_test_lstm, y_test_lstm = X_test, y_test

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(window_size, X_train_lstm.shape[2])))
lstm_model.add(tf.keras.layers.Dropout(0.2))
lstm_model.add(LSTM(32))
lstm_model.add(tf.keras.layers.Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

print("\nTraining LSTM model (this may take a little time)...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]
history = lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=200,
    batch_size=16,
    shuffle=False,
    callbacks=callbacks,
    verbose=1,
)


y_pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)

# Inverse-transform temperature only (feature index 0)
temp_min = scaler.data_min_[0]
temp_max = scaler.data_max_[0]
y_test_lstm_real = y_test_lstm * (temp_max - temp_min) + temp_min
y_pred_lstm_real = y_pred_lstm_scaled.flatten() * (temp_max - temp_min) + temp_min

lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_real, y_pred_lstm_real))
lstm_r2 = r2_score(y_test_lstm_real, y_pred_lstm_real)
print("\n=== LSTM Results ===")
print(f"RMSE: {lstm_rmse:.3f}")
print(f"R2  : {lstm_r2:.3f}")

# Dates aligned with test targets (same time indices as y_test_lstm)
test_time_index = target_time_index[test_mask]
test_dates_lstm = weather_df["date"].iloc[test_time_index]

plt.figure(figsize=(10, 4))
plt.plot(test_dates_lstm, y_test_lstm_real, label="Actual Temp")
plt.plot(test_dates_lstm, y_pred_lstm_real, label="LSTM Predicted Temp")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("LSTM Temperature Forecast")
plt.legend()
plt.tight_layout()
plt.show()


# --- Soil: columns match soil_data.csv ---
soil_df = pd.read_csv("soil_data.csv")

feature_cols = [
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
    "ph",
    "ec_ds_m",
    "organic_carbon_pct",
    "moisture_pct",
    "soil_type",
]
target_col = "label"

soil_enc = LabelEncoder()
soil_df = soil_df.copy()
soil_df["soil_type_code"] = soil_enc.fit_transform(soil_df["soil_type"].astype(str))

feature_cols_model = [c for c in feature_cols if c != "soil_type"] + ["soil_type_code"]
X_soil = soil_df[feature_cols_model]
y_soil = soil_df[target_col]

X_train_soil, X_test_soil, y_train_soil, y_test_soil = train_test_split(
    X_soil, y_soil, test_size=0.2, random_state=42, stratify=y_soil
)

dt_clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42,
)

dt_clf.fit(X_train_soil, y_train_soil)

y_pred_soil = dt_clf.predict(X_test_soil)

acc = accuracy_score(y_test_soil, y_pred_soil)
print("\n=== Decision Tree Soil Classification ===")
print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test_soil, y_pred_soil))

print("Confusion Matrix:")
print(confusion_matrix(y_test_soil, y_pred_soil))


def suggest_cultivation(temp_forecast, soil_label):
    """
    Rule-based suggestion using LSTM temperature and soil class.
    soil_label: Fertile / Moderate / Poor (from your soil_data.csv)
    """
    if soil_label == "Fertile" and 20 <= temp_forecast <= 35:
        return "Good conditions: Recommended to cultivate."
    if soil_label == "Moderate" and 18 <= temp_forecast <= 35:
        return "Moderate conditions: Cultivation possible with proper management."
    return "Not ideal: Consider improving soil or changing planting time."


example_soil_sample = pd.DataFrame(
    [
        {
            "nitrogen_ppm": 42,
            "phosphorus_ppm": 18,
            "potassium_ppm": 210,
            "ph": 6.4,
            "ec_ds_m": 0.42,
            "organic_carbon_pct": 0.68,
            "moisture_pct": 21.5,
            "soil_type": "Loamy",
        }
    ]
)
example_soil_sample["soil_type_code"] = soil_enc.transform(
    example_soil_sample["soil_type"].astype(str)
)
example_features = example_soil_sample[feature_cols_model]

predicted_soil_label = dt_clf.predict(example_features)[0]

example_temp_forecast = float(y_pred_lstm_real[-1])

decision = suggest_cultivation(example_temp_forecast, predicted_soil_label)
print("\n=== Decision Support Example ===")
print(f"Predicted soil class: {predicted_soil_label}")
print(f"Forecast temperature: {example_temp_forecast:.2f}°C")
print(f"Suggestion: {decision}")
