FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Expose HTTP port for validator ping
EXPOSE 8080

ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.1-70b-versatile"

# Health check — validator pings /reset, server.py handles it
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf -X POST http://localhost:8080/reset || exit 1

# Start HTTP ping server in background, then run inference
CMD ["sh", "-c", "python server.py & sleep 2 && python inference.py"]
