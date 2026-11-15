"""
ClimateOS – Deterministic WBGT Heat-Safety Engine for Singapore Schools
Public Hugging Face Space (FastAPI + Docker)
Version: 8.1.0 (LLM-free)
"""

import os
import re
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
import sklearn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_VERSION = "8.1.0"
SEASON_MAP = {
    "northeast_monsoon": [12, 1, 2, 3],
    "inter_monsoon_1": [4, 5],
    "southwest_monsoon": [6, 7, 8, 9],
    "inter_monsoon_2": [10, 11],
}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ClimateOS Core API",
    description="Deterministic school-specific WBGT heat-safety forecasts for Singapore",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Version check
try:
    with open("model_metadata.json") as f:
        model_metadata = json.load(f)
    trained_sklearn = model_metadata.get("sklearn_version")
    if trained_sklearn and trained_sklearn != sklearn.__version__:
        print(f"!!! VERSION MISMATCH !!! Trained with scikit-learn {trained_sklearn}, running {sklearn.__version__}.")
    else:
        print(f"[ClimateOS] scikit-learn {sklearn.__version__} matches model_metadata.json")
except Exception as e:
    print(f"[ClimateOS] WARNING: could not verify model_metadata.json ({e})")

# Seasonal residual experts
SEASONAL_EXPERTS: Dict[str, Any] = {}
print("[ClimateOS] Loading monsoonal residual experts...")
for season_name in SEASON_MAP:
    path = f"residual_model_{season_name}.pkl"
    if os.path.exists(path):
        try:
            SEASONAL_EXPERTS[season_name] = joblib.load(path)
            print(f"  → Mounted {season_name}")
        except Exception as e:
            print(f"  → Failed to load {path}: {e}")
    else:
        print(f"  → Missing {path}")

# Station offsets
try:
    with open("station_offsets.json") as f:
        station_offsets = json.load(f)
    print(f"[ClimateOS] Loaded {len(station_offsets)} station offsets")
except Exception as e:
    station_offsets = {}
    print(f"[ClimateOS] WARNING: station_offsets.json not loaded ({e})")

def _load_registry(path: str, label: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[ClimateOS] Loaded {label}: {len(data)} entries")
        return data
    except Exception as e:
        print(f"[ClimateOS] WARNING: {path} not loaded ({e})")
        return {}

school_registry = _load_registry("school_registry.json", "MOE schools")
venue_registry = _load_registry("venue_registry.json", "venues")
international_school_registry = _load_registry(
    "international_school_registry.json", "international schools"
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

PILOT_ALIASES = {
    # "RI": "raffles-institution",
}

def resolve_location(location_id: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    def _try(key: str):
        if key in school_registry:
            return "school", key, school_registry[key]
        if key in venue_registry:
            return "venue", key, venue_registry[key]
        if key in international_school_registry:
            return "international_school", key, international_school_registry[key]
        return None

    if location_id.upper() in PILOT_ALIASES:
        hit = _try(PILOT_ALIASES[location_id.upper()])
        if hit:
            return hit

    hit = _try(location_id)
    if hit:
        return hit

    hit = _try(slugify(location_id))
    if hit:
        return hit

    return None, None, None

def map_numeric_uv_to_who_band(uv_value) -> float:
    try:
        val = float(uv_value)
        if val <= 2.9: return 1.0
        elif val <= 5.9: return 2.0
        elif val <= 7.9: return 3.0
        elif val <= 10.9: return 4.0
        else: return 5.0
    except (ValueError, TypeError):
        return 2.0

def wbgt_physics(Ta: float, RH: float) -> float:
    e = (RH / 100.0) * 6.105 * np.exp((17.27 * Ta) / (237.7 + Ta))
    return 0.567 * Ta + 0.393 * e + 3.94

def predict_wbgt(
    Ta: float, RH: float, wind_speed: float, uv_band: float,
    hour: int, month: int, station_id: Optional[str] = None,
) -> float:
    phys = wbgt_physics(Ta, RH)

    assigned_regime = None
    for season_name, months in SEASON_MAP.items():
        if month in months:
            assigned_regime = season_name
            break

    if assigned_regime in SEASONAL_EXPERTS:
        features = pd.DataFrame([{
            "air_temp": Ta,
            "humidity": RH,
            "wind_speed": wind_speed,
            "uv_index_encoded": uv_band,
            "hour_of_day": hour,
            "month": month,
        }])
        correction = float(SEASONAL_EXPERTS[assigned_regime].predict(features)[0])
    else:
        correction = 0.0

    offset = station_offsets.get(station_id, 0.0) if station_id else 0.0
    return float(phys + correction + offset)

def evaluate_risk_tier(wbgt: float) -> Tuple[str, str]:
    if wbgt < 27.0:
        return "low", "Proceed as normal. Ensure water is available."
    elif wbgt < 31.0:
        return "moderate", "Increase water breaks; consider shaded areas; monitor vulnerable students closely."
    else:
        return "high", "Move session indoors or postpone; if unavoidable, sharply reduce intensity and duration."

def _select_period_forecast_text(periods: list, target_dt: datetime) -> str:
    for period in periods:
        try:
            start = datetime.fromisoformat(period["timePeriod"]["start"].replace("Z", "+00:00")).replace(tzinfo=None)
            end = datetime.fromisoformat(period["timePeriod"]["end"].replace("Z", "+00:00")).replace(tzinfo=None)
            if start <= target_dt < end:
                regions = period.get("regions", {})
                central = regions.get("central") or next(iter(regions.values()), {})
                return central.get("text", "Partly Cloudy")
        except (KeyError, ValueError, StopIteration):
            continue
    return "Partly Cloudy"

def estimate_future_uv(hour: int, forecast_text: str) -> float:
    if hour < 7 or hour > 19:
        return 0.0
    uv_bell_curve = {
        7: 0.0, 8: 1.2, 9: 3.5, 10: 6.2, 11: 9.0,
        12: 11.5, 13: 11.0, 14: 8.8, 15: 6.0,
        16: 3.2, 17: 1.0, 18: 0.1, 19: 0.0,
    }
    base_uv = uv_bell_curve.get(hour, 0.0)
    clean = forecast_text.lower()
    if any(kw in clean for kw in ["heavy", "thundery", "shower", "rain", "storm", "squall"]):
        multiplier = 0.20
    elif "cloudy" in clean or "overcast" in clean:
        multiplier = 0.50
    elif "partly cloudy" in clean:
        multiplier = 0.80
    else:
        multiplier = 1.00
    return base_uv * multiplier

def fetch_live_forecast_metrics(target_hour: int, target_dt: datetime) -> Tuple[float, float, float, float, str]:
    current_hour = datetime.now().hour
    temp, rh, wind = 31.5, 78.0, 8.0
    periods = []
    source = "seasonal_fallback_defaults"

    try:
        r = requests.get("https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast", timeout=8)
        if r.status_code == 200:
            rec = r.json()["data"]["records"][0]
            gen = rec["general"]
            temp = (float(gen["temperature"]["low"]) + float(gen["temperature"]["high"])) / 2
            rh = (float(gen["relativeHumidity"]["low"]) + float(gen["relativeHumidity"]["high"])) / 2
            wind = (float(gen["wind"]["speed"]["low"]) + float(gen["wind"]["speed"]["high"])) / 2
            periods = rec.get("periods", [])
            source = "live_nea_feed"
    except Exception:
        pass

    if target_hour == current_hour:
        try:
            r_uv = requests.get("https://api-open.data.gov.sg/v2/real-time/api/uv", timeout=8)
            if r_uv.status_code == 200:
                readings = r_uv.json().get("data", {}).get("records", [{}])[0].get("index", [])
                if readings:
                    uv_raw = float(readings[0].get("value", 4.5))
                    uv_band = map_numeric_uv_to_who_band(uv_raw)
                    return temp, rh, wind, uv_band, f"{source}_with_live_uv_nowcast"
        except Exception:
            pass

    forecast_text = _select_period_forecast_text(periods, target_dt)
    estimated_uv = estimate_future_uv(target_hour, forecast_text)
    uv_band = map_numeric_uv_to_who_band(estimated_uv)
    return temp, rh, wind, uv_band, f"{source}_with_predictive_solar_scaling"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "operational",
        "engine": f"ClimateOS Deterministic Chrono-Regime MoE ({APP_VERSION})",
        "active_experts": list(SEASONAL_EXPERTS.keys()),
        "microclimate_stations": len(station_offsets),
        "schools_registered": len(school_registry),
        "venues_registered": len(venue_registry),
        "international_schools_registered": len(international_school_registry),
        "llm_dependency": False,
    }

@app.get("/forecast/{location_id}")
def get_forecast(
    location_id: str,
    target_hour: Optional[int] = Query(None, description="Hour of day (0-23). Defaults to current hour."),
):
    now = datetime.now()
    if target_hour is None or not (0 <= target_hour <= 23):
        target_hour = now.hour
    target_dt = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

    loc_type, resolved_slug, entry = resolve_location(location_id)
    nearest_station = entry.get("nearest_wbgt_station") if entry else None
    display_name = None
    if entry:
        display_name = entry.get("school_name") or entry.get("venue_name")

    temp, rh, wind, uv_band, data_source = fetch_live_forecast_metrics(target_hour, target_dt)
    predicted_wbgt = predict_wbgt(temp, rh, wind, uv_band, target_hour, now.month, station_id=nearest_station)
    risk_tier, advisory = evaluate_risk_tier(predicted_wbgt)

    return {
        "queried_id": location_id,
        "target_hour_evaluated": target_hour,
        "resolved_slug": resolved_slug,
        "location_type": loc_type,
        "display_name": display_name or f"Unknown location ({location_id})",
        "nearest_station": nearest_station,
        "data_source": data_source,
        "metrics": {
            "air_temp_c": round(temp, 2),
            "humidity_pct": round(rh, 2),
            "wind_speed_kmh": round(wind, 2),
            "uv_who_ordinal_band": uv_band,
            "calculated_wbgt_c": round(predicted_wbgt, 2),
        },
        "evaluation": {
            "risk_level": risk_tier,
            "advisory_text": advisory,
        },
        "engine_version": APP_VERSION,
    }

@app.get("/search")
def search_locations(q: str = Query(..., min_length=2, description="Partial name or slug")):
    q_lower = q.lower().strip()
    results = []

    for slug, entry in school_registry.items():
        name = entry.get("school_name", "")
        if q_lower in name.lower() or q_lower in slug:
            results.append({"slug": slug, "name": name, "type": "school", "nearest_station": entry.get("nearest_wbgt_station")})

    for slug, entry in venue_registry.items():
        name = entry.get("venue_name", "")
        if q_lower in name.lower() or q_lower in slug:
            results.append({"slug": slug, "name": name, "type": "venue", "nearest_station": entry.get("nearest_wbgt_station")})

    for slug, entry in international_school_registry.items():
        name = entry.get("school_name", "")
        if q_lower in name.lower() or q_lower in slug:
            results.append({"slug": slug, "name": name, "type": "international_school", "nearest_station": entry.get("nearest_wbgt_station")})

    return {"query": q, "count": len(results), "results": results[:30]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
