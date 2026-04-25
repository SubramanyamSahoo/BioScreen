FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY bioscreen/ bioscreen/
COPY bioscreen-v1/ bioscreen-v1/

ENV BIOSCREEN_MODEL_PATH=/app/bioscreen-v1
ENV BIOSCREEN_API_KEY=""

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8080/v1/health || exit 1

CMD ["uvicorn", "bioscreen.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
