import numpy as np
import streamlit as st
import pandas as pd
import os
import mlflow


DATA_PATH = "smart_traffic_management_dataset.csv"
RAW_DATA = pd.read_csv(DATA_PATH)

PROCESSED_DATA = RAW_DATA.copy()
PROCESSED_DATA['timestamp'] = pd.to_datetime(PROCESSED_DATA['timestamp'])
PROCESSED_DATA['hour'] = PROCESSED_DATA['timestamp'].dt.hour
PROCESSED_DATA['day_of_week'] = PROCESSED_DATA['timestamp'].dt.dayofweek
PROCESSED_DATA['month'] = PROCESSED_DATA['timestamp'].dt.month
PROCESSED_DATA['is_weekend'] = (PROCESSED_DATA['day_of_week'] >= 5).astype(int)
PROCESSED_DATA['is_rush_hour'] = (
    ((PROCESSED_DATA['hour'] >= 7) & (PROCESSED_DATA['hour'] <= 9)) |
    ((PROCESSED_DATA['hour'] >= 17) & (PROCESSED_DATA['hour'] <= 19))
).astype(int)

FEATURE_DATA = pd.get_dummies(
    PROCESSED_DATA,
    columns=['weather_condition', 'signal_status'],
    drop_first=True,
    dtype=int,
)

TRAFFIC_FEATURES_DF = FEATURE_DATA.drop(columns=['traffic_volume', 'timestamp', 'location_id'])
ACCIDENT_FEATURES_DF = FEATURE_DATA.drop(columns=['accident_reported', 'timestamp', 'location_id'])

TRAFFIC_FEATURE_COLUMNS = TRAFFIC_FEATURES_DF.columns.tolist()
ACCIDENT_FEATURE_COLUMNS = ACCIDENT_FEATURES_DF.columns.tolist()

TRAFFIC_BASE_DEFAULTS = TRAFFIC_FEATURES_DF.mean(numeric_only=True).to_dict()
ACCIDENT_BASE_DEFAULTS = ACCIDENT_FEATURES_DF.mean(numeric_only=True).to_dict()

NUMERIC_CONTEXT_COLUMNS = [
    'traffic_volume',
    'avg_vehicle_speed',
    'vehicle_count_cars',
    'vehicle_count_trucks',
    'vehicle_count_bikes',
    'temperature',
    'humidity',
]


def _filter_with_fallback(df: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    subset = df[condition]
    return subset if not subset.empty else df


def get_context_stats(hour: int, day_of_week: int, month: int, weather: str, signal_status: str) -> dict:
    subset = PROCESSED_DATA
    subset = _filter_with_fallback(subset, subset['hour'] == hour)
    subset = _filter_with_fallback(subset, subset['day_of_week'] == day_of_week)
    subset = _filter_with_fallback(subset, subset['month'] == month)
    subset = _filter_with_fallback(subset, subset['weather_condition'] == weather)
    subset = _filter_with_fallback(subset, subset['signal_status'] == signal_status)

    stats = subset[NUMERIC_CONTEXT_COLUMNS].mean(numeric_only=True)
    fallback = PROCESSED_DATA[NUMERIC_CONTEXT_COLUMNS].mean(numeric_only=True)
    stats = stats.fillna(fallback)
    return stats.to_dict()

st.title("🚦 Smart Traffic Prediction Dashboard")

# Set MLflow tracking URI
if os.path.exists("/app/mlruns"):
    mlflow.set_tracking_uri("file:/app/mlruns")
else:
    mlflow.set_tracking_uri("file:./mlruns")

# Function to load ML models
@st.cache_resource
def load_xgboost_model():
    try:
        # Try to load from Model Registry
        model_uri = "models:/XGBoost_Forecaster/5"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load XGBoost model from Registry")
        st.info("💡 Note: Models work locally. In Docker, Model Registry paths may not resolve correctly.")
        return None

@st.cache_resource
def load_lightgbm_model():
    try:
        # Try to load from Model Registry
        model_uri = "models:/LightGBM_Accident_Predictor/5"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load LightGBM model from Registry")
        st.info("💡 Note: Models work locally. In Docker, Model Registry paths may not resolve correctly.")
        return None

# Check if MLflow experiments exist
has_models = os.path.exists("./mlruns") or os.path.exists("/app/mlruns")

if has_models:
    st.sidebar.success("✅ MLflow experiments detected")
else:
    st.sidebar.warning("⚠️ No MLflow experiments found. Run the notebook first.")

st.sidebar.header("🤖 Model Selection")
xgboost_run_id = st.sidebar.text_input("XGBoost Run ID (Traffic Volume)", "d463db149f03420ca019d6b4f5c37635")
lightgbm_run_id = st.sidebar.text_input("LightGBM Run ID (Accident Risk)", "cf33790947244d759d342d396c08f31f")

st.sidebar.header("📊 Prediction Type")
prediction_type = st.sidebar.radio(
    "Select Prediction Type",
    ["Traffic Volume", "Accident Risk"]
)

# Load appropriate model based on prediction type
if prediction_type == "Traffic Volume":
    model = load_xgboost_model()
else:
    model = load_lightgbm_model()

st.sidebar.header("🎯 Input Parameters")
hour = st.sidebar.slider("Hour of Day", 0, 23, 10)
day_of_week = st.sidebar.slider("Day of Week (Mon=0, Sun=6)", 0, 6, 1)
month = st.sidebar.slider("Month", 1, 12, 1)
weather_options = ["Cloudy", "Foggy", "Rainy", "Sunny", "Windy"]
weather = st.sidebar.selectbox("Weather Condition", weather_options, index=weather_options.index("Sunny"))
signal_status = st.sidebar.selectbox("Signal Status", ["Green", "Yellow", "Red"], index=0)


if st.sidebar.button(f"🔮 Predict {prediction_type}"):
    if not model:
        st.error("❌ Model not loaded. Cannot make predictions.")
        st.info("💡 Run this dashboard locally (outside Docker) to use the trained models.")
    else:
        try:
            if prediction_type == "Traffic Volume":
                input_data = TRAFFIC_BASE_DEFAULTS.copy()
                feature_columns = TRAFFIC_FEATURE_COLUMNS
            else:
                input_data = ACCIDENT_BASE_DEFAULTS.copy()
                feature_columns = ACCIDENT_FEATURE_COLUMNS

            weather_columns = [col for col in feature_columns if col.startswith('weather_condition_')]
            signal_columns = [col for col in feature_columns if col.startswith('signal_status_')]

            input_data.update({
                'hour': float(hour),
                'day_of_week': float(day_of_week),
                'month': float(month),
                'is_weekend': float(day_of_week >= 5),
                'is_rush_hour': float(hour in [7, 8, 9, 17, 18, 19]),
            })

            context_stats = get_context_stats(hour, day_of_week, month, weather, signal_status)
            for col, value in context_stats.items():
                if col in input_data:
                    input_data[col] = float(value)

            weather_mapping = {
                'Cloudy': None,
                'Foggy': 'weather_condition_Foggy',
                'Rainy': 'weather_condition_Rainy',
                'Sunny': 'weather_condition_Sunny',
                'Windy': 'weather_condition_Windy',
            }
            for col in weather_columns:
                if col in input_data:
                    input_data[col] = 0.0
            weather_column = weather_mapping.get(weather)
            if weather_column and weather_column in input_data:
                input_data[weather_column] = 1.0

            for col in signal_columns:
                if col in input_data:
                    input_data[col] = 0.0
            if signal_status == "Red" and 'signal_status_Red' in input_data:
                input_data['signal_status_Red'] = 1.0
            elif signal_status == "Yellow" and 'signal_status_Yellow' in input_data:
                input_data['signal_status_Yellow'] = 1.0

            input_df = pd.DataFrame([[input_data[col] for col in feature_columns]], columns=feature_columns)
            
            if prediction_type == "Traffic Volume":
                prediction = model.predict(input_df)
                predicted_volume = int(prediction[0])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Predicted Volume", value=f"{predicted_volume}")
                with col2:
                    congestion_level = "High" if predicted_volume > 600 else ("Medium" if predicted_volume > 300 else "Low")
                    st.metric(label="Congestion Level", value=congestion_level)
                with col3:
                    avg_speed = 60 - (predicted_volume / 20)
                    st.metric(label="Avg Speed (km/h)", value=f"{int(avg_speed)}")
                st.write("Input Features Used:")
                st.write(input_df)
            
            else:  
                st.subheader("⚠️ Accident Risk Assessment")
                raw_impl = getattr(model, "_model_impl", None)
                if raw_impl and hasattr(raw_impl, "lgb_model") and hasattr(raw_impl.lgb_model, "predict_proba"):
                    base_proba = raw_impl.lgb_model.predict_proba(input_df)[0][1]
                else:
                    preds = model.predict(input_df)
                    base_proba = float(preds[0] if np.ndim(preds) == 1 else preds[0][1])

                risk_adjustment = 0.0
                if weather in ["Rainy", "Foggy", "Windy"]:
                    risk_adjustment += 0.20
                if signal_status == "Red":
                    risk_adjustment += 0.25
                elif signal_status == "Yellow":
                    risk_adjustment += 0.15
                if hour in [7, 8, 9, 17, 18, 19]:
                    risk_adjustment += 0.15

                risk_proba = float(np.clip(base_proba + risk_adjustment, 0.0, 1.0))

                risk_score = int(risk_proba * 100)
                if risk_score >= 50:
                    risk_level = "🔴 HIGH"
                    recommendation = "⚠️ Alert traffic authorities. Consider speed limits and increased monitoring."
                elif risk_score >= 25:
                    risk_level = "🟡 MEDIUM"
                    recommendation = "⚡ Monitor conditions closely. Be prepared for incidents."
                else:
                    risk_level = "🟢 LOW"
                    recommendation = "✅ Normal operations. Continue standard monitoring."
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Risk Level", value=risk_level)
                with col2:
                    st.metric(label="Risk Score", value=f"{risk_score}%")
                
                st.markdown(f"**Recommendation:** {recommendation}")
                st.markdown("### Risk Factors:")
                factors = []
                if risk_score > 30:
                    factors.append("• High accident probability based on model prediction")
                if weather in ["Rainy", "Foggy", "Windy", "Cloudy"]:
                    factors.append(f"• Adverse weather: {weather}")
                if hour in [7, 8, 9, 17, 18, 19]:
                    factors.append(f"• Rush hour: {hour}:00")
                if signal_status in ["Red", "Yellow"]:
                    factors.append(f"• Signal status caution: {signal_status}")
                
                if factors:
                    for factor in factors:
                        st.markdown(factor)
                else:
                    st.markdown("• No significant risk factors detected")
                st.write("Input Features Used:")
                st.write(input_df)
                st.caption(f"Base model probability: {base_proba:.2%} | Adjusted probability: {risk_proba:.2%}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Debug info:")
            st.write(input_df.columns.tolist())
        
st.subheader("Raw Project Data")
st.dataframe(RAW_DATA)


