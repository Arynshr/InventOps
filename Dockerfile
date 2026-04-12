FROM python:3.11-slim
 
WORKDIR /app
 
# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
 
# Install Python deps — no cache to avoid stale layers
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
 
# Copy entire project
COPY . .
 
# Install package
RUN pip install --no-cache-dir -e .
 
# Sanity check — fail build early if critical files or imports are missing
RUN python -c "from openai import OpenAI; print('openai OK')"
RUN python -c "from InventOps import SupplyChainEnv; print('InventOps OK')"
RUN test -f /app/server.py   || (echo "ERROR: server.py missing"   && exit 1)
RUN test -f /app/inference.py || (echo "ERROR: inference.py missing" && exit 1)
 
EXPOSE 8080
 
ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.1-8b-instant"
 
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf -X POST http://localhost:8080/reset || exit 1
 
# Start server, wait for it to bind, then run inference
CMD ["sh", "-c", "python server.py & sleep 2 && python inference.py"]
