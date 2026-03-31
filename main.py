

import os
import logging
import numpy as np
import pandas as pd
import requests
import joblib

from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH      = "disaster_weather_model.pkl"
OPEN_METEO_LIVE = "https://api.open-meteo.com/v1/forecast"

SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}

SEVERITY_ADVICE = {
    0: "Low risk — standard monitoring recommended.",
    1: "Medium risk — prepare response teams and issue advisories.",
    2: "High risk — immediate evacuation and emergency response required.",
}

WEATHER_FALLBACK = {
    "temperature_2m_max":         28.0,
    "precipitation_sum":          15.0,
    "windspeed_10m_max":          20.0,
    "pressure_msl_mean":         1010.0,
    "relativehumidity_2m_mean":   70.0,
}

VALID_DISASTER_TYPES = [
    "Flood",
    "Earthquake",
    "Storm",
    "Volcanic activity",
    "Mass movement (wet)",
]

VALID_REGIONS = [
    "Southern Asia",
    "South-Eastern Asia",
    "Eastern Asia",
    "Western Asia",
    "Africa",
    "Americas",
    "Europe",
    "Oceania",
]

app = FastAPI(
    title="Disaster Severity Prediction API",
    description=(
        "Predicts disaster severity (Low / Medium / High) for any location "
        "using real-time weather from Open-Meteo and a trained ML model. "
        "Designed for Flutter app integration."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["GET"],
    allow_headers=["*"],
)


model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file '{MODEL_PATH}' not found. Train the model first.")
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found.")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from '{MODEL_PATH}'")



def _first(lst):
    """Safely return first element of a list, or None."""
    if lst and len(lst) > 0:
        val = lst[0]
        return None if (val is None or (isinstance(val, float) and np.isnan(val))) else val
    return None


def fetch_live_weather(lat: float, lon: float) -> dict:
    """
    Fetches today's weather at lat/lon from Open-Meteo forecast API.
    Falls back to training-dataset medians if the API is unreachable.
    Adds a 'source' key: 'live' or 'fallback'.
    """
    try:
        params = {
            "latitude":      round(lat, 4),
            "longitude":     round(lon, 4),
            "daily": [
                "temperature_2m_max",
                "precipitation_sum",
                "windspeed_10m_max",
                "pressure_msl_mean",
                "relativehumidity_2m_mean",
            ],
            "forecast_days": 1,
            "timezone":      "auto",
        }
        r = requests.get(OPEN_METEO_LIVE, params=params, timeout=10)
        r.raise_for_status()
        daily = r.json().get("daily", {})

        weather = {
            "temperature_2m_max":       _first(daily.get("temperature_2m_max")),
            "precipitation_sum":        _first(daily.get("precipitation_sum")),
            "windspeed_10m_max":        _first(daily.get("windspeed_10m_max")),
            "pressure_msl_mean":        _first(daily.get("pressure_msl_mean")),
            "relativehumidity_2m_mean": _first(daily.get("relativehumidity_2m_mean")),
        }
        
        for key, fallback_val in WEATHER_FALLBACK.items():
            if weather[key] is None:
                weather[key] = fallback_val

        weather["source"] = "live"
        return weather

    except Exception as e:
        logger.warning(f"Open-Meteo unavailable ({e}) — using fallback weather values.")
        return {**WEATHER_FALLBACK, "source": "fallback"}


def build_feature_row(
    lat:              float,
    lon:              float,
    disaster_type:    str,
    disaster_subtype: str,
    country:          str,
    region:           str,
    weather:          dict,
    duration_days:    int = 1,
) -> pd.DataFrame:
    """
    Builds a single-row DataFrame matching the exact feature set
    the model was trained on in Section 4 of disaster_prediction.py.
    """
    precip   = weather.get("precipitation_sum",        0)    or 0
    wind     = weather.get("windspeed_10m_max",         0)    or 0
    temp     = weather.get("temperature_2m_max",        25)   or 25
    pressure = weather.get("pressure_msl_mean",         1013) or 1013
    humidity = weather.get("relativehumidity_2m_mean",  60)   or 60

    is_heavy_rain    = int(precip   > 50)
    is_high_wind     = int(wind     > 60)
    is_low_pressure  = int(pressure < 990)
    is_extreme_heat  = int(temp     > 40)
    is_high_humidity = int(humidity > 85)
    climate_risk     = sum([
        is_heavy_rain, is_high_wind,
        is_low_pressure, is_extreme_heat, is_high_humidity,
    ])

    row = {
        "Disaster Type":    disaster_type,
        "Disaster Subtype": disaster_subtype,
        "Country":          country,
        "Region":           region,

        "Magnitude_log":         0.0,
        "Sum of Latitude":       lat,
        "Sum of Longitude":      lon,
        "disaster_duration":     max(duration_days / 30, 0),
        "disaster_month":        datetime.today().month,
        "Injured_log":           0.0,
        "Homeless_log":          0.0,
        "country_avg_sev":       1.0,
        "type_avg_sev":          1.0,

        "temperature_2m_max":         temp,
        "precipitation_sum":          precip,
        "windspeed_10m_max":          wind,
        "pressure_msl_mean":          pressure,
        "relativehumidity_2m_mean":   humidity,

        "precip_log":         np.log1p(precip),
        "wind_log":           np.log1p(wind),
        "is_heavy_rain":      is_heavy_rain,
        "is_high_wind":       is_high_wind,
        "is_low_pressure":    is_low_pressure,
        "is_extreme_heat":    is_extreme_heat,
        "is_high_humidity":   is_high_humidity,
        "climate_risk_score": climate_risk,
    }
    return pd.DataFrame([row])



@app.get("/", tags=["Health"])
def root():
    """
    Health check.
    Call this from Flutter on app launch to confirm the API is alive.

    Example Flutter call:
        final res = await http.get(Uri.parse('http://YOUR_IP:8000/'));
    """
    return {
        "status":       "ok",
        "message":      "Disaster Severity Prediction API is running.",
        "version":      "1.0.0",
        "model_loaded": model is not None,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }


@app.get("/predict", tags=["Prediction"])
def predict(
    lat: float = Query(
        ...,
        description="Latitude of the disaster location",
        example=28.6139,
    ),
    lon: float = Query(
        ...,
        description="Longitude of the disaster location",
        example=77.2090,
    ),
    disaster_type: str = Query(
        ...,
        description="Type of disaster (must match a trained type)",
        example="Flood",
    ),
    country: str = Query(
        ...,
        description="Country name",
        example="India",
    ),
    region: str = Query(
        ...,
        description="Geographic region",
        example="Southern Asia",
    ),
    disaster_subtype: str = Query(
        "Unknown",
        description="Disaster subtype — optional",
        example="Flash flood",
    ),
    duration_days: int = Query(
        1,
        description="Expected event duration in days",
        example=3,
    ),
):
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Restart the server.")

    # ── Validate coordinates ─────────────────────────────────────────
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180.")
    if duration_days < 0:
        raise HTTPException(status_code=422, detail="duration_days cannot be negative.")

    # ── Warn on unknown disaster type (don't reject — model handles it) ──
    type_warning = None
    if disaster_type not in VALID_DISASTER_TYPES:
        type_warning = (
            f"'{disaster_type}' was not in training data. "
            f"Valid types: {VALID_DISASTER_TYPES}. Prediction confidence may be lower."
        )
        logger.warning(type_warning)

    weather = fetch_live_weather(lat, lon)

    input_df = build_feature_row(
        lat=lat, lon=lon,
        disaster_type=disaster_type,
        disaster_subtype=disaster_subtype,
        country=country,
        region=region,
        weather=weather,
        duration_days=duration_days,
    )

    try:
        pred_class = int(model.predict(input_df)[0])
        pred_proba = model.predict_proba(input_df)[0].tolist()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    confidence    = round(pred_proba[pred_class] * 100, 1)
    low_conf_flag = confidence < 60

    climate_flags = {
        "heavy_rain":    weather.get("precipitation_sum",       0)    > 50,
        "high_wind":     weather.get("windspeed_10m_max",        0)    > 60,
        "low_pressure":  weather.get("pressure_msl_mean",       1013) < 990,
        "extreme_heat":  weather.get("temperature_2m_max",       25)   > 40,
        "high_humidity": weather.get("relativehumidity_2m_mean", 60)   > 85,
    }
    climate_risk_score = sum(climate_flags.values())

    response = {
        "severity_code":  pred_class,
        "severity_label": SEVERITY_LABELS[pred_class],
        "advice":         SEVERITY_ADVICE[pred_class],

        "confidence_pct": confidence,
        "low_confidence": low_conf_flag,
        "probabilities": {
            "Low":    round(pred_proba[0] * 100, 1),
            "Medium": round(pred_proba[1] * 100, 1),
            "High":   round(pred_proba[2] * 100, 1),
        },

        "weather": {
            "temperature_c": weather.get("temperature_2m_max"),
            "rainfall_mm":   weather.get("precipitation_sum"),
            "wind_kmh":      weather.get("windspeed_10m_max"),
            "pressure_hpa":  weather.get("pressure_msl_mean"),
            "humidity_pct":  weather.get("relativehumidity_2m_mean"),
            "data_source":   weather.get("source"),
        },

        "climate_risk_score": climate_risk_score,
        "climate_flags":      climate_flags,

        "request": {
            "latitude":       lat,
            "longitude":      lon,
            "disaster_type":  disaster_type,
            "country":        country,
            "region":         region,
            "duration_days":  duration_days,
        },

        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if type_warning:
        response["warning"] = type_warning

    return response


@app.get("/weather", tags=["Weather"])
def get_weather(
    lat: float = Query(..., description="Latitude",  example=28.6139),
    lon: float = Query(..., description="Longitude", example=77.2090),
):
    
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180.")

    weather = fetch_live_weather(lat, lon)

    return {
        "latitude":      lat,
        "longitude":     lon,
        "temperature_c": weather.get("temperature_2m_max"),
        "rainfall_mm":   weather.get("precipitation_sum"),
        "wind_kmh":      weather.get("windspeed_10m_max"),
        "pressure_hpa":  weather.get("pressure_msl_mean"),
        "humidity_pct":  weather.get("relativehumidity_2m_mean"),
        "data_source":   weather.get("source"),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    
    return {
        "model_file":            MODEL_PATH,
        "model_loaded":          model is not None,
        "model_type":            type(model.named_steps["model"]).__name__ if model else None,
        "severity_classes":      SEVERITY_LABELS,
        "accuracy_note":         "Overall accuracy ~65% on held-out test set (3-class problem).",
        "valid_disaster_types":  VALID_DISASTER_TYPES,
        "valid_regions":         VALID_REGIONS,
        "weather_source":        "Open-Meteo (open-meteo.com) — free, no API key required",
        "timestamp":             datetime.utcnow().isoformat() + "Z",
    }


@app.get("/disaster-types", tags=["Reference"])
def disaster_types():

    return {
        "disaster_types": VALID_DISASTER_TYPES,
        "regions":        VALID_REGIONS,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)