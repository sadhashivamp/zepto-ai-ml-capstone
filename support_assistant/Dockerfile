FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant ./support_assistant

RUN python support_assistant/ingest.py

CMD ["uvicorn", "support_assistant.api.main:app", "--host", "0.0.0.0", "--port", "7860"]