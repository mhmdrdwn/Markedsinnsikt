FROM python:3.11-slim

WORKDIR /app

# ARM64-native Chromium for kaleido (chart image export)
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium \
 && rm -rf /var/lib/apt/lists/*

# Tell kaleido/choreographer to use the apt Chromium instead of downloading AMD64 Chrome
ENV BROWSER_PATH=/usr/bin/chromium

# Install deps first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /usr/local/lib/python3.11/site-packages/choreographer/cli/browser_exe/

COPY . .

# Pre-compile all .py files so first-request import is faster
RUN python -m compileall -q .

CMD ["gunicorn", "app.main:server", \
     "--workers", "1", \
     "--threads", "4", \
     "--bind", "0.0.0.0:8050", \
     "--timeout", "120", \
     "--preload"]
