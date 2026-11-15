"""
ClimateOS – Deterministic School-Specific WBGT Heat-Safety Engine (Singapore)
Gradio Space version (free tier compatible)
Version: 8.2.0 – Fully deterministic, no LLM
"""

import os
import re
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
import joblib
import sklearn
import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_VERSION = "8.2.0"
SEASON_MAP = {
    "northeast_monsoon": [12, 1, 2, 3],
    "inter_monsoon_1": [4, 5],
    "southwest_monsoon": [6, 7, 8, 9],
    "inter_monsoon_2": [10, 11],
}

print(f"[ClimateOS] Starting v{APP_VERSION} (Gradio, deterministic)")

try:
    with open("model_metadata.json") as f:
        model_metadata = json.load(f)
    trained_sklearn = model_metadata.get("sklearn_version")
    if trained_sklearn and trained_sklearn != sklearn.__version__:
        print(f"!!! VERSION MISMATCH !!! Trained {trained_sklearn} vs running {sklearn.__version__}")
    else:
        print(f"[ClimateOS] scikit-learn {sklearn.__version__} OK")
except Exception as e:
    print(f"[ClimateOS] model_metadata.json note: {e}")

SEASONAL_EXPERTS: Dict[str, Any] = {}
for season_name in SEASON_MAP:
    path = f"residual_model_{season_name}.pkl"
    if os.path.exists(path):
        try:
            SEASONAL_EXPERTS[season_name] = joblib.load(path)
            print(f"  → Mounted {season_name}")
        except Exception as e:
            print(f"  → Failed {path}: {e}")
    else:
        print(f"  → Missing {path}")

try:
    with open("station_offsets.json") as f:
        station_offsets = json.load(f)
    print(f"[ClimateOS] {len(station_offsets)} station offsets loaded")
except Exception as e:
    station_offsets = {}
    print(f"[ClimateOS] station_offsets.json missing: {e}")

def _load_registry(path: str, label: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[ClimateOS] {label}: {len(data)} entries")
        return data
    except Exception as e:
        print(f"[ClimateOS] {path} missing: {e}")
        return {}

school_registry = _load_registry("school_registry.json", "MOE schools")
venue_registry = _load_registry("venue_registry.json", "venues")
international_school_registry = _load_registry(
    "international_school_registry.json", "international schools"
)

ALL_LOCATIONS: List[Dict[str, str]] = []
for slug, entry in school_registry.items():
    ALL_LOCATIONS.append({
        "slug": slug,
        "name": entry.get("school_name", slug),
        "type": "School (MOE)",
        "station": entry.get("nearest_wbgt_station", ""),
    })
for slug, entry in venue_registry.items():
    ALL_LOCATIONS.append({
        "slug": slug,
        "name": entry.get("venue_name", slug),
        "type": "Venue",
        "station": entry.get("nearest_wbgt_station", ""),
    })
for slug, entry in international_school_registry.items():
    ALL_LOCATIONS.append({
        "slug": slug,
        "name": entry.get("school_name", slug),
        "type": "International School",
        "station": entry.get("nearest_wbgt_station", ""),
    })

print(f"[ClimateOS] Total searchable locations: {len(ALL_LOCATIONS)}")

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

def resolve_location(location_id: str):
    def _try(key: str):
        if key in school_registry:
            return "school", key, school_registry[key]
        if key in venue_registry:
            return "venue", key, venue_registry[key]
        if key in international_school_registry:
            return "international_school", key, international_school_registry[key]
        return None

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

def predict_wbgt(Ta, RH, wind_speed, uv_band, hour, month, station_id=None) -> float:
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

def evaluate_risk_tier(wbgt: float):
    if wbgt < 27.0:
        return "Low", "Proceed as normal. Ensure water is available."
    elif wbgt < 31.0:
        return "Moderate", "Increase water breaks; consider shaded areas; monitor vulnerable students closely."
    else:
        return "High", "Move session indoors or postpone; if unavoidable, sharply reduce intensity and duration."

def _select_period_forecast_text(periods, target_dt):
    for period in periods:
        try:
            start = datetime.fromisoformat(period["timePeriod"]["start"].replace("Z", "+00:00")).replace(tzinfo=None)
            end = datetime.fromisoformat(period["timePeriod"]["end"].replace("Z", "+00:00")).replace(tzinfo=None)
            if start <= target_dt < end:
                regions = period.get("regions", {})
                central = regions.get("central") or next(iter(regions.values()), {})
                return central.get("text", "Partly Cloudy")
        except Exception:
            continue
    return "Partly Cloudy"

def estimate_future_uv(hour: int, forecast_text: str) -> float:
    if hour < 7 or hour > 19:
        return 0.0
    uv_bell = {
        7: 0.0, 8: 1.2, 9: 3.5, 10: 6.2, 11: 9.0,
        12: 11.5, 13: 11.0, 14: 8.8, 15: 6.0,
        16: 3.2, 17: 1.0, 18: 0.1, 19: 0.0,
    }
    base = uv_bell.get(hour, 0.0)
    clean = forecast_text.lower()
    if any(k in clean for k in ["heavy", "thundery", "shower", "rain", "storm", "squall"]):
        mult = 0.20
    elif "cloudy" in clean or "overcast" in clean:
        mult = 0.50
    elif "partly cloudy" in clean:
        mult = 0.80
    else:
        mult = 1.00
    return base * mult

def fetch_live_forecast_metrics(target_hour: int, target_dt: datetime):
    current_hour = datetime.now().hour
    temp, rh, wind = 31.5, 78.0, 8.0
    periods = []
    source = "seasonal_fallback_defaults"

    try:
        r = requests.get(
            "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast",
            timeout=8,
        )
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
                    return temp, rh, wind, map_numeric_uv_to_who_band(uv_raw), f"{source}_with_live_uv_nowcast"
        except Exception:
            pass

    forecast_text = _select_period_forecast_text(periods, target_dt)
    estimated_uv = estimate_future_uv(target_hour, forecast_text)
    return temp, rh, wind, map_numeric_uv_to_who_band(estimated_uv), f"{source}_with_predictive_solar_scaling"

def search_locations(query: str):
    if not query or len(query.strip()) < 2:
        return gr.Dropdown(choices=[], value=None)

    q = query.lower().strip()
    matches = []
    for loc in ALL_LOCATIONS:
        if q in loc["name"].lower() or q in loc["slug"]:
            label = f"{loc['name']}  ({loc['type']})"
            matches.append(label)
            if len(matches) >= 25:
                break
    return gr.Dropdown(choices=matches, value=matches[0] if matches else None)

def run_forecast(selected_label: str, target_hour: int):
    if not selected_label:
        return (
            "—", "—", "—",
            "Please search and select a school or venue first.",
            "—", "—", "—", "—", "—"
        )

    name = selected_label.split("  (")[0].strip()

    entry = None
    slug = None
    for loc in ALL_LOCATIONS:
        if loc["name"] == name:
            slug = loc["slug"]
            break

    if not slug:
        _, slug, entry = resolve_location(name)
    else:
        _, _, entry = resolve_location(slug)

    if not entry:
        return (
            "—", "—", "—",
            f"Could not resolve location: {name}",
            "—", "—", "—", "—", "—"
        )

    nearest_station = entry.get("nearest_wbgt_station")
    display_name = entry.get("school_name") or entry.get("venue_name") or name

    now = datetime.now()
    if target_hour is None or not (0 <= int(target_hour) <= 23):
        target_hour = now.hour
    target_hour = int(target_hour)
    target_dt = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

    temp, rh, wind, uv_band, data_source = fetch_live_forecast_metrics(target_hour, target_dt)
    wbgt = predict_wbgt(temp, rh, wind, uv_band, target_hour, now.month, station_id=nearest_station)
    risk_level, advisory = evaluate_risk_tier(wbgt)

    risk_display = {
        "Low": "🟢 LOW",
        "Moderate": "🟠 MODERATE",
        "High": "🔴 HIGH",
    }.get(risk_level, risk_level)

    return (
        f"{wbgt:.1f} °C",
        risk_display,
        advisory,
        f"{display_name}\nNearest station: {nearest_station or 'N/A'}  |  Data: {data_source}",
        f"{temp:.1f} °C",
        f"{rh:.0f} %",
        f"{wind:.1f} km/h",
        f"{uv_band:.0f}",
        f"{target_hour:02d}:00",
    )

with gr.Blocks(
    title="ClimateOS – Singapore School Heat Safety",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
) as demo:

    gr.Markdown(
        f"""
        # 🌡️ ClimateOS – School Heat Safety (Singapore)
        **Deterministic WBGT forecast for PE teachers & sports coordinators**  
        Version {APP_VERSION} · No LLM · Physics + seasonal residual experts
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            search_box = gr.Textbox(
                label="Search school or venue",
                placeholder="Type at least 2 letters, e.g. raffles, anglo, nus, bedok...",
                lines=1,
            )
            location_dropdown = gr.Dropdown(
                label="Select location",
                choices=[],
                interactive=True,
                allow_custom_value=False,
            )
            hour_slider = gr.Slider(
                minimum=6,
                maximum=19,
                step=1,
                value=datetime.now().hour if 6 <= datetime.now().hour <= 19 else 14,
                label="Target hour (local time)",
            )
            predict_btn = gr.Button("Get Heat-Safety Forecast", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("### Forecast Result")
            wbgt_out = gr.Textbox(label="Predicted WBGT", interactive=False)
            risk_out = gr.Textbox(label="Risk Level", interactive=False)
            advisory_out = gr.Textbox(label="Advisory", interactive=False, lines=2)
            meta_out = gr.Textbox(label="Location & Data Source", interactive=False, lines=2)

            with gr.Row():
                temp_out = gr.Textbox(label="Air Temp", interactive=False)
                rh_out = gr.Textbox(label="Humidity", interactive=False)
                wind_out = gr.Textbox(label="Wind", interactive=False)
                uv_out = gr.Textbox(label="UV Band (WHO)", interactive=False)
                hour_out = gr.Textbox(label="Hour Evaluated", interactive=False)

    gr.Markdown(
        """
        ---
        **How to use**  
        1. Type part of a school or venue name  
        2. Select the correct location from the dropdown  
        3. Choose the hour of the PE / CCA session  
        4. Click **Get Heat-Safety Forecast**

        **Risk bands (adapted for exertional activity)**  
        - 🟢 **Low** (< 27 °C WBGT) – Proceed as normal  
        - 🟠 **Moderate** (27–31 °C) – Extra water breaks, shade, monitor students  
        - 🔴 **High** (≥ 31 °C) – Prefer indoors or postpone; sharply reduce intensity if outdoor is unavoidable
        """
    )

    search_box.change(fn=search_locations, inputs=search_box, outputs=location_dropdown)
    predict_btn.click(
        fn=run_forecast,
        inputs=[location_dropdown, hour_slider],
        outputs=[wbgt_out, risk_out, advisory_out, meta_out, temp_out, rh_out, wind_out, uv_out, hour_out],
    )

demo.queue(default_concurrency_limit=5)
demo.launch()
