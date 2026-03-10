FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy model and configs
COPY outputs_v3/best_model.pth outputs_v3/
COPY outputs_v3/temperature.json outputs_v3/
COPY outputs_v3/thresholds.json outputs_v3/
COPY data/fundus_norm_stats.json data/

# Copy API code
COPY api/main.py api/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
