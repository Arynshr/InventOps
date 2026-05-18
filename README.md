# InventOps — Supply Chain RL Environment

> OpenEnv-compliant reinforcement learning environment for supply chain optimization
> with a built-in model visibility dashboard powered by Streamlit + SQLite.

---

## Overview

InventOps simulates a inventory management problem where an LLM agent
must issue sequential `order / transfer / hold` decisions to minimize stockouts,
holding costs, and capacity breaches across a planning horizon.

**Stack:** Python 3.11 · Pydantic v2 · NumPy · Groq API · Streamlit · Plotly · SQLite  
**Tasks:** `easy` / `medium` / `hard`  
**Reward:** Dense shaped (fulfillment − holding − stockout − capacity penalties)

---

## Project Structure

```text
InventOps/
├── InventOps/          # Core RL environment (env, models, reward, simulator)
├── rlvr/               # RLVR loop — GroqAgent + PromptOptimizer
│   └── prompts/        # Base & optimised prompt text files
├── metrics/            # SQLite metric logger (auto-created on first run)
│   └── logger.py       # MetricLogger — thread-safe, no-op-capable
├── dashboard/          # Streamlit visibility dashboard
│   ├── app.py          # 4-tab UI (Overview · Episodes · RLVR · Inference)
│   └── queries.py      # SQL → pandas query helpers
├── inference.py        # HF/OpenEnv submission entry-point
├── evaluate.py         # Multi-agent benchmark (hold / random / LLM)
├── server.py           # FastAPI action server
├── Dockerfile          # Main inference image
├── Dockerfile.dashboard# Lightweight dashboard image
└── docker-compose.yml  # Full stack (inference + dashboard, shared DB volume)
````

---

## Quick Start

### 1 — Install

```bash
# Recommended: uv (fast)
uv sync

# Or pip
pip install -r requirements.txt
```

---

### 2 — Run evaluations (no API key needed)

```bash
# Hold-only + random baselines, 10 seeds
python evaluate.py --seeds 10
```

Include Groq LLM agent:

```bash
GROQ_API_KEY=gsk_... python evaluate.py --seeds 10 --llm
```

---

### 3 — Run inference (LLM submission)

```bash
HF_TOKEN=gsk_... python inference.py
```

Self-test without API key:

```bash
python inference.py --test
```

---

### 4 — Run the RLVR prompt optimiser

```bash
GROQ_API_KEY=gsk_... python rlvr/prompt_optimizer.py --task medium --rounds 4
```

---

### 5 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

→ [http://localhost:8501](http://localhost:8501)

---

## Model Visibility Dashboard

All three entry-points (`inference.py`, `evaluate.py`, `rlvr/prompt_optimizer.py`)
automatically write structured metrics to `metrics/inventops.db` (SQLite).

The Streamlit dashboard reads from this file and provides four tabs:

| Tab              | What you see                                                               |
| ---------------- | -------------------------------------------------------------------------- |
| **🏠 Overview**  | KPI cards · Mean score by task & agent · Recent runs table                 |
| **📈 Episodes**  | Reward-per-step curves · Action distribution pie · Reward components       |
| **🔁 RLVR Loop** | Score progression per prompt round · min/max band · Failure type breakdown |
| **⚡ Inference** | LLM latency histogram · Parse error rate · Raw step log                    |

---

## Docker

### Full stack (inference server + dashboard)

```bash
# Copy .env.example → .env and fill in keys
docker compose up --build
```

Services:

* `inventops` → [http://localhost:8080](http://localhost:8080)  (FastAPI action server)
* `dashboard` → [http://localhost:8501](http://localhost:8501)  (Streamlit dashboard)

The two containers share a named Docker volume (`metrics_data`) so the dashboard
updates live as inference runs write step data.

---

### Dashboard only

```bash
docker build -f Dockerfile.dashboard -t inventops-dashboard .

docker run -p 8501:8501 \
  -v $(pwd)/metrics:/app/metrics \
  inventops-dashboard
```

---

## Environment Variables

| Variable       | Description                                 | Default                          |
| -------------- | ------------------------------------------- | -------------------------------- |
| `HF_TOKEN`     | Groq / HF / OpenRouter API key              | —                                |
| `GROQ_API_KEY` | Groq key (used by rlvr/ and evaluate --llm) | —                                |
| `API_BASE_URL` | LLM endpoint                                | `https://api.groq.com/openai/v1` |
| `MODEL_NAME`   | Model identifier                            | `llama-3.1-8b-instant`           |
| `INVENTOPS_DB` | Path to SQLite metrics database             | `metrics/inventops.db`           |

---

## Evaluation

```text
=======================================================================
  InventOps — Benchmark Evaluation  (20 seeds per task)
=======================================================================
Task        hold-only              random             groq-llm
              mean ± std           mean ± std          mean ± std
-------------------------------------------------------------------------
easy        0.412 ± 0.091       0.389 ± 0.103       0.631 ± 0.072
medium      0.388 ± 0.087       0.401 ± 0.098       0.584 ± 0.081
hard        0.341 ± 0.094       0.362 ± 0.110       0.547 ± 0.089
-------------------------------------------------------------------------
composite       0.380               0.384               0.587
=======================================================================
```

---

## License

Apache License 2.0
