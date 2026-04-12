FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

# Sanity checks
RUN python -c "from openai import OpenAI; print('openai OK')"
RUN python -c "from fastapi import FastAPI; print('fastapi OK')"
RUN python -c "from InventOps import SupplyChainEnv; print('InventOps OK')"
RUN test -f /app/server.py    || (echo "ERROR: server.py missing"    && exit 1)
RUN test -f /app/inference.py || (echo "ERROR: inference.py missing" && exit 1)

# HF Spaces requires port 7860
EXPOSE 7860

ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.1-8b-instant"

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Start FastAPI server in background, wait for it, then run inference once
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 7860 & SERVER_PID=$! && sleep 3 && python inference.py; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null"]
