---
title: InventOps
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# InventOps — Supply Chain RL Environment

## Overview
OpenEnv-compliant reinforcement learning environment for supply chain optimization.

**Stack:** Python 3.11, Pydantic v2, NumPy, Groq API  
**Tasks:** Easy / Medium / Hard  
**Reward:** Dense shaped  

## Usage
```bash
# Run inference
HF_TOKEN=gsk_... python inference.py

# Self-test (no API key needed)
python inference.py --test

# Run evaluator
python evaluate.py --seeds 20
```

## Environment Variables
| Variable | Description |
|---|---|
| `HF_TOKEN` | Your Groq / HF / OpenRouter API key |
| `API_BASE_URL` | LLM endpoint (default: Groq) |
| `MODEL_NAME` | Model identifier |
