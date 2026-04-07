# InventOps — Implementation Plan

## Overview

**Goal:** Build an OpenEnv-compliant reinforcement learning environment for supply chain optimization with 3 tasks, reward shaping, and a baseline agent.

**Stack:** Python 3.11, Pydantic v2, NumPy, Groq API

---

## 1. Architecture

```
inventops/
├── env.py          # Main environment (reset, step, state, grade)
├── simulator.py    # State transitions
├── models.py       # Data schemas
├── reward.py       # Reward function
├── demand.py       # Demand generator
├── tasks/          # Easy / Medium / Hard
├── data/           # Config + demand profiles
├── baseline.py     # Agent script
├── evaluate.py     # Evaluation logic
```

---

## 2. Core Flow

```
reset()
  → initialize simulator
  → return observation

step(action)
  → validate action
  → apply action
  → process arrivals
  → generate demand
  → update inventory
  → compute reward
  → return (obs, reward, done, info)

grade()
  → return score ∈ [0, 1]
```

---

## 3. Core Components

### Models (`models.py`)

* Observation
* Action (order / transfer / hold)
* StepReward (detailed reward)
* EpisodeInfo (logging)

---

### Simulator (`simulator.py`)

Handles:

* Inventory updates
* Order lifecycle (lead times)
* Demand fulfillment
* Supplier disruptions
* Capacity constraints

**Owns all mutable state**

---

### Demand (`demand.py`)

* Stochastic demand (normal / poisson)
* Seasonality + trend
* Rolling forecast

---

### Reward (`reward.py`)

Pure function:

**Reward =**

* * Fulfillment
* − Holding cost
* − Stockouts
* − Order & transfer cost
* − Capacity violations
* − Bullwhip penalty

---

### Environment (`env.py`)

Implements:

* `reset()`
* `step()`
* `state()`
* `grade()`

Acts as wrapper over simulator.

---

## 4. Tasks

### Easy

* Single SKU, single warehouse
* Objective: minimize stockouts and cost

### Medium

* Multiple SKUs + budget constraint
* Objective: maintain high fill rate with cost control

### Hard

* Multi-warehouse network
* Includes supplier disruptions and routing complexity

---

## 5. Grading

Each task returns score in **[0, 1]**

Based on:

* Fill rate / stockouts
* Cost efficiency vs baseline
* Constraint violations

**Must be deterministic**

---

## 8. Design Principles

* **Pure reward function** → easy testing
* **Simulator owns state** → clean separation
* **Strong typing first** → fewer bugs
* **Seeded randomness** → reproducibility

---
