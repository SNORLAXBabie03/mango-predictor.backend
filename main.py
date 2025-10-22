# ==========================================================
# 🌾 Mango Profit Prediction Backend (Hybrid Forecast + Archive)
# ==========================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import os

# ----------------------------------------------------------
# 1️ Setup Flask
# ----------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------
# 2️ Load Trained Model
# ----------------------------------------------------------
model = xgb.XGBRegressor()
model.load_model("xgb_model_v1.json")  # Must match your trained model filename

# ----------------------------------------------------------
# 3️ Fetch Weather Data (Forecast or Historical)
# ----------------------------------------------------------
def fetch_weather_data(induction_date, harvest_date, latitude=15.3491, longitude=119.9668):
    """Fetch weather data between induction and harvest dates.
       If harvest date > today → use 16-day forecast.
       Else → use historical archive data.
    """
    induction_dt = datetime.strptime(induction_date, "%Y-%m-%d")
    harvest_dt = datetime.strptime(harvest_date, "%Y-%m-%d")
    today = datetime.now()

    # ✅ Ensure harvest is always after induction
    if harvest_dt <= induction_dt:
        harvest_dt = induction_dt + timedelta(days=120)

    print(f"🌦 Fetching weather from {induction_dt.date()} to {harvest_dt.date()}")

    # --- Setup API session ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ----------------------------------------------------------
    # 🌤️ Case 1: Future harvest date → Use 16-day forecast
    # ----------------------------------------------------------
    if harvest_dt > today:
        print("📈 Using 16-day forecast data (future dates detected).")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum"],
            "forecast_days": 16,
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()

        data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "max_temp": daily.Variables(0).ValuesAsNumpy(),
            "min_temp": daily.Variables(1).ValuesAsNumpy(),
            "rainfall(mm)": daily.Variables(2).ValuesAsNumpy(),
        }
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["source"] = "forecast"
        print("✅ 16-Day Forecast Retrieved")
        return df

    # ----------------------------------------------------------
    # 🌦️ Case 2: Past/Current harvest date → Use archive data
    # ----------------------------------------------------------
    else:
        print("📜 Using historical archive data (past dates).")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": induction_dt.strftime("%Y-%m-%d"),
            "end_date": harvest_dt.strftime("%Y-%m-%d"),
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum"],
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()

        data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "max_temp": daily.Variables(0).ValuesAsNumpy(),
            "min_temp": daily.Variables(1).ValuesAsNumpy(),
            "rainfall(mm)": daily.Variables(2).ValuesAsNumpy(),
        }
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["source"] = "historical"
        print("✅ Historical Weather Retrieved")
        return df


# ----------------------------------------------------------
# 4️ Aggregate Weather Data
# ----------------------------------------------------------
def aggregate_weather(weather_df):
    """Compute average and total weather metrics."""
    if weather_df.empty:
        return [np.nan]*5

    avg_temp = (weather_df["min_temp"] + weather_df["max_temp"]) / 2

    return [
        float(avg_temp.mean()),                   # avg_temp
        float(weather_df["rainfall(mm)"].mean()), # avg_rainfall
        float(weather_df["rainfall(mm)"].sum()),  # total_rainfall
        float(weather_df["min_temp"].min()),      # min_temp_stage
        float(weather_df["max_temp"].max())       # max_temp_stage
    ]

# ----------------------------------------------------------
# 5️ Define Feature Order (Must match training)
# ----------------------------------------------------------
BASE_FEATURES = [
    "induced_trees", "spraying_frequency", "labor_worker", "working_days",
    "revenue", "growing_days", "total_yield",
    "avg_temp", "avg_rainfall", "total_rainfall",
    "fertilizer_No", "fertilizer_Yes",
    "process_Chemical_Induction", "process_Natural", "process_Palusot"
]

# ----------------------------------------------------------
# 6️ Routes
# ----------------------------------------------------------
@app.route("/")
def home():
    return "✅ Flask backend is running and ready for mango profit predictions!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📩 Received:", data)

        if not data:
            return jsonify({"error": "No input data received"}), 400

        # --- Extract Inputs ---
        induction_start = str(data.get("inductionStart", "2024-01-01"))
        harvest_date = str(data.get("harvestDate", "2024-05-01"))
        trees_induced = float(data.get("treesInduced", 0))
        spraying_frequency = float(data.get("sprayingFrequency", 0))
        labor_worker = float(data.get("workers", 0))
        working_days = float(data.get("workingDays", 0))
        revenue = float(data.get("revenue", 0))
        total_yield = float(data.get("totalYield", 0))
        fertilizer_usage = str(data.get("fertilizerUsage", "No"))
        production_process = str(data.get("productionProcess", "Chemical_Induction"))

        # --- Compute growing days ---
        induction_dt = datetime.strptime(induction_start, "%Y-%m-%d").replace(year=2024)
        harvest_dt = datetime.strptime(harvest_date, "%Y-%m-%d")
        growing_days = (harvest_dt - induction_dt).days
        growing_days = max(0, min(growing_days, 150))  # cap at 150 days

        # --- One-hot encode fertilizer ---
        fertilizer_encoded = [
            1.0 if fertilizer_usage == "No" else 0.0,
            1.0 if fertilizer_usage == "Yes" else 0.0
        ]

        # --- One-hot encode production process ---
        process_encoded = [
            1.0 if production_process == "Chemical_Induction" else 0.0,
            1.0 if production_process == "Natural" else 0.0,
            1.0 if production_process == "Palusot" else 0.0
        ]

        # --- Fetch and aggregate weather ---
        weather_df = fetch_weather_data(induction_start, harvest_date)
        avg_temp, avg_rainfall, total_rainfall, _, _ = aggregate_weather(weather_df)

        # --- Build feature vector ---
        features = [
            trees_induced, spraying_frequency, labor_worker, working_days,
            revenue, growing_days, total_yield,
            avg_temp, avg_rainfall, total_rainfall,
            *fertilizer_encoded, *process_encoded
        ]

        X_input = pd.DataFrame([features], columns=BASE_FEATURES).astype(np.float32)

        print("\n🧮 Final Feature Vector:")
        print(X_input.to_dict(orient="records")[0])

        # --- Predict ---
        prediction = model.predict(X_input)
        net_profit = float(prediction[0])
        print(f"💰 Predicted Net Profit: ₱{net_profit:,.2f}")

        return jsonify({
            "net_profit": net_profit,
            "features": X_input.to_dict(orient="records")[0],
            "weather": {
                "avg_temp": avg_temp,
                "avg_rainfall": avg_rainfall,
                "total_rainfall": total_rainfall,
                "period_start": weather_df["date"].min().strftime("%Y-%m-%d"),
                "period_end": weather_df["date"].max().strftime("%Y-%m-%d"),
                "source": weather_df["source"].iloc[0] if "source" in weather_df else "unknown"
            }
        })

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------
# 7️ Run Server
# ----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
