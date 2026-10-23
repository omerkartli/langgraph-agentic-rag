FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY data ./data
# The index is built on startup with the active embedding provider (see app/api.py).
ENV CHROMA_PATH=/srv/chroma_db
ENV PORT=8080
CMD exec uvicorn app.api:app --host 0.0.0.0 --port $PORT
