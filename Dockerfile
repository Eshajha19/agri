FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --force-reinstall python-dotenv

RUN pip install gunicorn

# Copy source files
COPY . .

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build
WORKDIR /app

EXPOSE 8000

ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["bash", "-c", "gunicorn main:app -w ${WEB_CONCURRENCY:-1} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}"]