---

title: InventOps
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
-------------

# InventOps — Supply Chain RL Environment

> A production-grade, OpenEnv-compliant reinforcement learning environment for **inventory optimization and supply chain decision-making**.

---

## 🚀 Overview

InventOps simulates a **multi-warehouse, multi-SKU supply chain** under real-world constraints:

* Stochastic demand (seasonality + trend)
* Supplier disruptions
* Lead times & pending orders
* Warehouse capacity limits
* Budget constraints (medium task)

It is designed for:

* RL researchers
* LLM agent builders
* Supply chain engineers

---

## 🧠 Key Capabilities

### 📦 Environment

* Multi-SKU, multi-warehouse simulation
* Deterministic seeding for reproducibility
* Structured observations (Pydantic models)

### 📊 Demand Modeling

* Normal & Poisson distributions
* Weekly seasonality
* Long-term trend factors
* Rolling forecasts

### ⚙️ Constraints

* Supplier availability / disruptions
* Lead time distributions
* Warehouse capacity limits
* Budget constraints (task-dependent)

### 💰 Reward Function

Dense shaped reward composed of:

* Fulfillment revenue
* Holding cost penalty
* Stockout penalty (2× multiplier)
* Order costs (fixed + variable)
* Transfer costs
* Capacity breach penalty
* Bullwhip effect penalty

---

## 🎯 Tasks

| Task     | Description                               | Episode Length |
| -------- | ----------------------------------------- | -------------- |
| `easy`   | Single SKU reorder optimization           | 30             |
| `medium` | Multi-SKU + budget constraints            | 60             |
| `hard`   | Full network with transfers & disruptions | 90             |

Each task includes a **custom grading function** that evaluates:

* Service level
* Cost efficiency
* Operational behavior

---

## 🏗️ Architecture

```
InventOps/
  env.py            # RL interface (reset, step, grade)
  simulator.py      # Core simulation engine
  demand.py         # Demand generation
  reward.py         # Reward computation
  models.py         # Typed schemas (Pydantic)
  tasks/            # Task-specific grading logic

rlvr/
  agent.py          # LLM agent wrapper
  prompt_optimizer.py
  prompts/

server/
  app.py            # FastAPI interface

tests/              # Full validation suite
```

---

## ⚡ Quick Start

### 1. Install

```bash
git clone <repo-url>
cd inventops

pip install -r requirements.txt
pip install -e .
```

---

### 2. Run Inference

```bash
HF_TOKEN=your_api_key python inference.py
```

---

### 3. Run Self-Test (No API Key Needed)

```bash
python inference.py --test
```

---

### 4. Run Benchmark

```bash
python evaluate.py --seeds 20
```

---

## 🤖 LLM Agent

The system includes a built-in LLM agent with:

* Observation → prompt formatting
* JSON action parsing with fallback
* Error handling (invalid JSON → `hold`)

### Example Action

```json
{
  "action_type": "order",
  "sku_id": "SKU_01",
  "quantity": 50,
  "target_warehouse": "WH_1"
}
```

---

## 📡 API Server

Start the environment server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Endpoints

| Endpoint  | Description            |
| --------- | ---------------------- |
| `/health` | Health check           |
| `/reset`  | Initialize environment |
| `/step`   | Execute action         |
| `/state`  | Raw simulator state    |
| `/schema` | Observation schema     |

---

## 🐳 Docker

```bash
docker build -t inventops .
docker run -p 7860:7860 inventops
```

Includes:

* API server
* Inference runner
* Health checks

---

## 🧪 Testing

```bash
pytest tests/
```

Covers:

* Environment correctness
* Reward consistency
* Determinism
* API contract validation
* LLM fallback safety

---

## 📊 Evaluation

### Baseline

```bash
python baseline.py --seeds 5
```

### Full Evaluation

```bash
python evaluate.py --seeds 20
```

Outputs:

* Mean score per task
* Standard deviation
* Composite score

---

## 🧠 Prompt Optimization (RLVR)

Automatically improves LLM prompts:

```bash
python -m rlvr.prompt_optimizer --rounds 4
```

Produces:

* Optimized prompt
* Performance logs
* Failure analysis

---

## 📄 Repomix Packed File (Important)

This repository may include a **Repomix-packed file** with the following properties:

* Merged representation of selected repository files
* Comments and empty lines removed
* Code compressed using `⋮----` delimiters
* Security checks disabled

### ⚠️ Usage Guidelines

* Treat packed files as **read-only**
* Modify original source files instead
* Use file headers (`## File: path`) to distinguish content
* Handle carefully — may contain sensitive information

---

## ⚙️ Configuration

Defined in:

```
openenv.yaml
```

Includes:

* Task definitions
* Episode lengths
* Dependencies
* OpenEnv metadata

---

## 🛡️ Design Principles

* Deterministic evaluation (seed-controlled)
* Strict schema validation (Pydantic)
* Fail-safe execution (invalid actions → `hold`)
* Modular simulation engine
* LLM-first design

---

## 📈 Future Work

* Multi-echelon supplier networks
* Dynamic pricing integration
* Real-time demand sensing
* Advanced RL baselines (PPO, SAC)
* Distributed simulation

---

## 📜 License

Apache License 2.0

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Submit a PR

---

## 🏁 Summary

InventOps is a **realistic, extensible, and production-ready environment** for:

* Reinforcement learning research
* LLM-based decision systems
* Supply chain optimization

If you're building **autonomous logistics intelligence**, this is your sandbox.
