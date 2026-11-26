# services/ingestion/Dockerfile - build from repository root.
# Copies both services/ingestion/src and ml/src so online_features.py and
# schemas.py compile against ml/src/aegis_ml/features/definitions.py.
FROM python:3.12-slim

WORKDIR /app

COPY services/ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ingestion/src /app/services/ingestion/src
COPY ml/src /app/ml/src

ENV PYTHONPATH=/app/services/ingestion/src:/app/ml/src \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD python -c \
    "import urllib.request as u; u.urlopen('http://localhost:8001/health', timeout=3)"

CMD ["uvicorn", "ingestion.main:app", "--host", "0.0.0.0", "--port", "8001"]
