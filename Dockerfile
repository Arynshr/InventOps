FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8080
ENV PYTHONPATH=/app

# Default: run baseline evaluation with 3 seeds (fast smoke test)
CMD ["python", "baseline.py", "--seeds", "3"]
