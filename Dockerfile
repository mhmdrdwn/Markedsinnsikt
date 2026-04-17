FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-compile .py files so first-request import is faster
RUN python -m compileall -q .

CMD ["gunicorn", "app.main:server", \
     "--workers", "1", \
     "--threads", "4", \
     "--bind", "0.0.0.0:8050", \
     "--timeout", "120", \
     "--preload"]
