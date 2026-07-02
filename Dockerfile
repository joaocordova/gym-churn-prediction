# ---------------------------------------------------------------------------
# gym-churn-prediction — containerized scoring dashboard
#
#   docker build -t gym-churn .
#   docker run -p 8501:8501 gym-churn
#
# The image ships the trained model + generated assets, so it serves
# predictions immediately. To retrain inside the container instead:
#   docker run gym-churn python -m gym_churn.cli all
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required by LightGBM's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer-cache dependencies separately from source.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY configs/ configs/
COPY src/ src/
RUN pip install --no-deps .

COPY app.py .
COPY models/ models/
COPY assets/ assets/
COPY data/sample/ data/sample/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
