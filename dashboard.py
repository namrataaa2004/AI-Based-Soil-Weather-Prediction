# ============================================================
# Dashboard: Soil & Weather Conditions for Cultivation
# Aligned with soil_only.py (RandomForest + soil_data.csv) and
# main.py (weather_data.csv: ARIMA + LSTM + decision rules)
# ============================================================
import streamlit as st                                           
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

st.set_page_config(
    page_title="Soil & Weather Cultivation Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom style
st.markdown(
    """
<style>
    .main-header { font-size: 1.8rem; color: #2d5a27; margin-bottom: 0.5rem; }
    .metric-card { background: #f0f7ee; padding: 1rem; border-radius: 8px; border-left: 4px solid #2d5a27; margin: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">🌾 Prediction of Soil and Weather Conditions for Cultivation</p>',
    unsafe_allow_html=True,
)
st.markdown("Using Data Science, Data Mining, Analysis & Artificial Intelligence")
st.divider()

# --- Data loaders (same CSVs as main.py / soil_only.py) ---
@st.cache_data
def load_weather():
    try:
        df = pd.read_csv("weather_data.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None

@st.cache_data
def load_soil():
    try:
        return pd.read_csv("soil_data.csv")
    except Exception:
        return None

# --- Soil: same logic as soil_only.py (RandomForest + encoded soil_type) ---
# NOTE: models/encoders should be cached as "resource", not "data".
@st.cache_resource
def train_soil_model():
    soil_df = load_soil()
    if soil_df is None or len(soil_df) < 4:
        return None, None, None, None, None

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
    missing = [c for c in feature_cols + ["label"] if c not in soil_df.columns]
    if missing:
        return None, None, None, None, None

    df = soil_df.copy()
    le = LabelEncoder()
    df["soil_type_code"] = le.fit_transform(df["soil_type"].astype(str))
    feature_cols_model = [c for c in feature_cols if c != "soil_type"] + ["soil_type_code"]
    X = df[feature_cols_model]
    y = df["label"]

    # soil_only.py uses random_state=42, no stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test, le, feature_cols_model

def normalize_soil_label(label) -> str:
    """
    Map dataset-specific labels into the 3 decision-rule classes:
    Fertile / Moderate / Poor.
    """
    if label is None:
        return "Poor"
    s = str(label).strip().lower()
    if "fert" in s or "good" in s or "high" in s:
        return "Fertile"
    if "moder" in s or "medium" in s or "avg" in s:
        return "Moderate"
    if "poor" in s or "low" in s or "bad" in s:
        return "Poor"
    # fallback: if label is already one of them (different casing)
    if s in {"fertile", "moderate", "poor"}:
        return s.title()
    return "Poor"

def suggest_cultivation(temp_forecast, soil_label):
    """Same rules as main.py (labels: Fertile / Moderate / Poor)."""
    if soil_label == "Fertile" and 20 <= temp_forecast <= 35:
        return "Good conditions: Recommended to cultivate."
    if soil_label == "Moderate" and 18 <= temp_forecast <= 35:
        return "Moderate conditions: Cultivation possible with proper management."
    return "Not ideal: Consider improving soil or changing planting time."

def create_sequences(values, window_size=7):
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🌤️ Weather", "🌱 Soil Classification", "✅ Decision Support"]
)

with tab1:
    st.subheader("Project Overview")
    weather_df = load_weather()
    soil_df = load_soil()
    c1, c2, c3 = st.columns(3)
    with c1:
        if weather_df is not None:
            st.metric("Weather records", len(weather_df))
            st.caption("weather_data.csv")
        else:
            st.metric("Weather records", "—")
    with c2:
        if soil_df is not None:
            st.metric("Soil samples", len(soil_df))
            st.caption("soil_data.csv")
        else:
            st.metric("Soil samples", "—")
    with c3:
        if weather_df is not None and "temperature_c" in weather_df.columns:
            avg_temp = weather_df["temperature_c"].mean()
            st.metric("Avg temperature (°C)", f"{avg_temp:.1f}")
        else:
            st.metric("Avg temperature", "—")
    st.info(
        "Use **Weather** for ARIMA/LSTM (same as main.py), **Soil** for Random Forest "
        "(same as soil_only.py), and **Decision Support** for cultivation advice."
    )

with tab2:
    st.subheader("Weather Data & Temperature Forecast (main.py style)")
    weather_df = load_weather()
    if weather_df is None:
        st.warning(
            "Could not load weather_data.csv. Place it next to dashboard.py "
            "(columns: date, temperature_c, humidity_pct, …)."
        )
    elif "temperature_c" not in weather_df.columns:
        st.error("weather_data.csv must include a **temperature_c** column.")
    else:
        st.dataframe(weather_df, use_container_width=True, hide_index=True)
        chart_cols = ["temperature_c"]
        if "humidity_pct" in weather_df.columns:
            chart_cols.append("humidity_pct")
        st.line_chart(weather_df.set_index("date")[chart_cols])

        col_a, col_b = st.columns(2)
        with col_a:
            run_arima = st.button("Run ARIMA forecast", type="primary")
        with col_b:
            run_lstm = st.button("Run LSTM forecast (TensorFlow)")

        temp_series = weather_df["temperature_c"].astype(float).values

        if run_arima:
            try:
                from statsmodels.tsa.arima.model import ARIMA

                split_index = max(1, int(len(temp_series) * 0.8))
                if split_index >= len(temp_series):
                    split_index = len(temp_series) - 1
                train_temp = temp_series[:split_index]
                test_temp = temp_series[split_index:]
                if len(train_temp) < 3 or len(test_temp) < 1:
                    st.warning("Not enough rows for train/test split. Add more dates.")
                else:
                    p = min(5, max(1, len(train_temp) - 2))
                    arima_order = (p, 1, 0)
                    model = ARIMA(train_temp, order=arima_order)
                    result = model.fit()
                    arima_forecast = result.forecast(steps=len(test_temp))
                    rmse = np.sqrt(mean_squared_error(test_temp, arima_forecast))
                    r2 = r2_score(test_temp, arima_forecast) if len(test_temp) > 1 else 0.0
                    st.success(
                        f"ARIMA {arima_order} — RMSE: {rmse:.3f}  |  R²: {r2:.3f}"
                    )
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(
                        weather_df["date"].iloc[split_index:],
                        test_temp,
                        label="Actual",
                    )
                    ax.plot(
                        weather_df["date"].iloc[split_index:],
                        arima_forecast,
                        label="ARIMA Predicted",
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("ARIMA Temperature Forecast")
                    ax.legend()
                    ax.tick_params(axis="x", rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"ARIMA failed: {e}")

        if run_lstm:
            try:
                import tensorflow as tf

                tf.get_logger().setLevel("ERROR")
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense

                window_size = 7
                if len(temp_series) <= window_size + 5:
                    st.warning(
                        f"Need more than {window_size + 5} rows for LSTM windowing."
                    )
                else:
                    with st.spinner("Training LSTM (may take a minute)…"):
                        scaler = MinMaxScaler()
                        temp_scaled = scaler.fit_transform(
                            temp_series.reshape(-1, 1)
                        ).flatten()
                        X_all, y_all = create_sequences(temp_scaled, window_size)
                        split_i = int(len(X_all) * 0.8)
                        X_train = X_all[:split_i].reshape(-1, window_size, 1)
                        X_test = X_all[split_i:].reshape(-1, window_size, 1)
                        y_train = y_all[:split_i]
                        y_test = y_all[split_i:]

                        lstm_model = Sequential(
                            [
                                LSTM(64, input_shape=(window_size, 1)),
                                Dense(1),
                            ]
                        )
                        lstm_model.compile(optimizer="adam", loss="mse")
                        lstm_model.fit(
                            X_train,
                            y_train,
                            epochs=20,
                            batch_size=16,
                            validation_split=0.1,
                            verbose=0,
                        )
                        y_pred_s = lstm_model.predict(X_test, verbose=0)
                        y_test_real = scaler.inverse_transform(
                            y_test.reshape(-1, 1)
                        ).flatten()
                        y_pred_real = scaler.inverse_transform(y_pred_s).flatten()

                    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
                    r2 = r2_score(y_test_real, y_pred_real)
                    st.success(f"LSTM — RMSE: {rmse:.3f}  |  R²: {r2:.3f}")

                    dates_for_y = weather_df["date"].iloc[window_size:].reset_index(
                        drop=True
                    )
                    test_dates = dates_for_y.iloc[
                        split_i : split_i + len(y_test_real)
                    ]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(test_dates, y_test_real, label="Actual")
                    ax.plot(test_dates, y_pred_real, label="LSTM Predicted")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("LSTM Temperature Forecast")
                    ax.legend()
                    ax.tick_params(axis="x", rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.session_state["lstm_last_forecast"] = float(y_pred_real[-1])
            except ImportError:
                st.error("TensorFlow is not installed. Run: pip install tensorflow")
            except Exception as e:
                st.error(f"LSTM failed: {e}")

with tab3:
    st.subheader("Soil Classification — Random Forest (soil_only.py style)")
    soil_df = load_soil()
    if soil_df is not None:
        st.dataframe(soil_df, use_container_width=True, hide_index=True)
        clf, X_test, y_test, le, _ = train_soil_model()
        if clf is not None:
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Model accuracy (Random Forest)", f"{acc:.2%}")
            st.text("Classification Report")
            st.code(classification_report(y_test, y_pred))
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
        else:
            st.warning(
                "Could not train model: check soil_data.csv columns "
                "(nitrogen_ppm, phosphorus_ppm, potassium_ppm, ph, ec_ds_m, "
                "organic_carbon_pct, moisture_pct, soil_type, label)."
            )
    else:
        st.warning("Could not load soil_data.csv.")

with tab4:
    st.subheader("Cultivation Decision Support (main.py rules)")
    clf, _, _, le, feature_cols_model = train_soil_model()
    if clf is not None and le is not None and feature_cols_model is not None:
        st.write(
            "Enter soil readings (same units as soil_data.csv). "
            "Then click **Predict** to update the result."
        )

        soil_types = list(le.classes_)
        has_lstm = "lstm_last_forecast" in st.session_state

        with st.form("decision_support_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                n_ppm = st.number_input("Nitrogen (ppm)", 0, 400, 42, key="ds_n")
                p_ppm = st.number_input("Phosphorus (ppm)", 0, 50, 18, key="ds_p")
                k_ppm = st.number_input("Potassium (ppm)", 0, 300, 210, key="ds_k")
                ph = st.slider("pH", 4.0, 9.0, 6.4, 0.1, key="ds_ph")
            with c2:
                ec = st.slider("EC (dS/m)", 0.0, 2.0, 0.42, 0.01, key="ds_ec")
                oc = st.slider("Organic carbon (%)", 0.0, 1.5, 0.68, 0.01, key="ds_oc")
                moisture_pct = st.slider("Moisture (%)", 0.0, 50.0, 21.5, 0.5, key="ds_m")
                soil_type = st.selectbox(
                    "Soil type",
                    soil_types,
                    index=0,
                    key="ds_soil_type",
                )

            use_lstm = st.checkbox(
                "Use last LSTM forecast as temperature (run LSTM in Weather tab first)",
                value=has_lstm,
                disabled=not has_lstm,
                key="ds_use_lstm",
            )
            if use_lstm and has_lstm:
                temp = float(st.session_state["lstm_last_forecast"])
                st.caption(f"Using LSTM forecast: **{temp:.2f}°C**")
            else:
                if not has_lstm:
                    st.caption("Run **LSTM forecast** in the Weather tab to enable this option.")
                temp = st.slider("Temperature (°C)", 15.0, 40.0, 26.0, 0.5, key="ds_temp")

            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            try:
                code = le.transform([soil_type])[0]
            except ValueError:
                st.error("Invalid soil type for encoder.")
                st.stop()

            sample = pd.DataFrame(
                [
                    {
                        "nitrogen_ppm": n_ppm,
                        "phosphorus_ppm": p_ppm,
                        "potassium_ppm": k_ppm,
                        "ph": ph,
                        "ec_ds_m": ec,
                        "organic_carbon_pct": oc,
                        "moisture_pct": moisture_pct,
                        "soil_type_code": code,
                    }
                ]
            )[feature_cols_model]

            raw_pred_label = clf.predict(sample)[0]
            pred_label = normalize_soil_label(raw_pred_label)
            suggestion = suggest_cultivation(temp, pred_label)
            st.success(f"**Predicted soil class:** {pred_label}")
            st.caption(f"Model raw label: {raw_pred_label}")
            st.info(f"**Suggestion:** {suggestion}")
            st.caption(f"Temperature used: {temp:.2f}°C")
    else:
        st.warning(
            "Train the soil model by placing soil_data.csv next to dashboard.py "
            "(see Soil Classification tab)."
        )

st.divider()
st.caption(
    "Prediction of Soil and Weather Conditions for Cultivation — Data Science, Mining, Analysis & AI"
)
