---
title: ClimateOS Singapore School Heat Safety
emoji: 🌡️
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Deterministic school-specific WBGT forecasts for Singapore PE teachers
---

# ClimateOS – School-Specific WBGT Heat-Safety Engine (Singapore)

## The problem
Singapore PE teachers and sports coordinators must decide daily whether outdoor physical activity is safe under high heat and humidity. Island-wide weather apps give a single temperature or general forecast; they do not account for micro-climate differences between schools or the actual Wet Bulb Globe Temperature (WBGT) that governs heat-stress risk. Wrong decisions either expose students to heat illness or cancel sessions unnecessarily.

## What we built
ClimateOS is a deterministic API that takes a school (or public sports venue) identifier and an optional target hour, then returns:
- Predicted WBGT (°C) at that location and hour
- Risk tier (low / moderate / high) mapped to Singapore-adapted advisory language
- Supporting meteorology (air temperature, humidity, wind, UV band)

Teachers query a simple REST endpoint (or a thin front-end built on top of it) and receive an actionable, location-specific recommendation without any large-language-model involvement.

## What we added
Our own idea. Built from public NEA / data.gov.sg real-time and historical feeds, MOE school directory, SportSG facilities, and a curated international-school list. No starter template was used.

## How it works

**Data**  
- Historical WBGT station readings, air temperature, relative humidity, wind speed and UV index pulled from data.gov.sg (v1 environment endpoints + v2 real-time APIs).  
- Stratified sampling across ~12 months (5 days per month) yielded several thousand matched station-hour records after cleaning and deduplication.  
- School and venue locations geocoded via OneMap and matched by haversine distance to the nearest active WBGT station. Failed geocodes were logged rather than fabricated.

**Model**  
Physics-informed residual mixture-of-experts:  
1. A closed-form outdoor WBGT approximation supplies the baseline.  
2. Four GradientBoostingRegressor experts (one per Singapore monsoon regime) learn the residual (actual station WBGT − physics estimate) from air temperature, humidity, wind, UV WHO band, hour and month.  
3. Station-specific offsets capture persistent micro-climate bias.  
Out-of-fold cross-validation within each regime prevents leakage. UV for future hours is estimated from a deterministic solar-elevation curve attenuated by the time-matched NEA forecast text—never by persisting a stale reading.

## Results
On held-out out-of-fold residuals the model with UV features reduced residual standard deviation relative to the physics-only baseline and relative to an ablated no-UV model (exact figures available in the training notebook). Residual errors remain largest under rapidly changing cloud cover and at stations that experienced network expansion during the training window. The system therefore surfaces the data-source tag (`live_nea_feed` vs `seasonal_fallback_defaults`) so users can judge confidence.

## What we'd do next
1. Add a short (6–12 h) temporal smoother that blends successive forecast calls, reducing hour-to-hour jitter caused by discrete forecast-period boundaries.  
2. Collect a small set of on-campus WBGT logger readings from pilot schools to quantify and further correct remaining station-to-campus bias.

## Sources
- NEA / data.gov.sg real-time and historical environment APIs: https://data.gov.sg and https://api-open.data.gov.sg  
- MOE “General Information of Schools” dataset  
- SportSG facilities dataset  
- OneMap Singapore geocoding API: https://www.onemap.gov.sg  
- Scikit-learn GradientBoostingRegressor  
- Training and evaluation notebook (Climate_Veritas) developed in Google Colab  

**API quick start**  
