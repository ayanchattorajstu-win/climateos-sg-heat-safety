FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_metadata.json .
COPY station_offsets.json .
COPY school_registry.json .
COPY venue_registry.json .
COPY international_school_registry.json .
COPY residual_model_northeast_monsoon.pkl .
COPY residual_model_inter_monsoon_1.pkl .
COPY residual_model_southwest_monsoon.pkl .
COPY residual_model_inter_monsoon_2.pkl .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
