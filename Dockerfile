# Imagen multi-stage: dependencias pesadas (xgboost, mlflow, etc.) se
# instalan en `base` y no se reconstruyen si solo cambia el código en `src/`.

FROM python:3.11-slim AS base
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS production
COPY src/ ./src/
COPY configs/ ./configs/
COPY params.yaml ./params.yaml
COPY main.py ./main.py
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
