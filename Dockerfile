FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/
COPY scripts/ scripts/

ENV MODEL_DIR=/srv/models
ENV PREDICTION_LOG=/srv/data/prediction_log.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
