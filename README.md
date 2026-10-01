# Credit Resiliency Intelligence

A personal learning prototype exploring how supervised machine learning and a contextual bandit could support personalized treatment decisions for credit-card customers experiencing financial hardship.

> **Important:** This project uses synthetic data and simulated rewards. It is a learning prototype, not a production treatment policy or evidence of real-world treatment effectiveness.

## Business Problem

Customers experiencing financial hardship may respond differently to available treatments such as payment plans, hardship programs, or settlement offers.

This project demonstrates a simple two-stage approach:

1. **Predict default risk** with an XGBoost classifier.
2. **Recommend a treatment** with a LinUCB contextual bandit using customer characteristics as context.

The goal is to explore the difference between predicting customer risk and choosing an action that may improve an outcome.

## Approach

### 1. Default Risk Prediction

The supervised-learning component uses XGBoost to estimate customer default probability from synthetic customer, credit, delinquency, and hardship features.

Model evaluation includes standard classification diagnostics such as ROC-AUC and confusion-matrix analysis.

### 2. Treatment Recommendation

The main treatment-learning experiment uses a **LinUCB contextual bandit**.

- **Context:** customer characteristics and predicted default risk
- **Actions:** payment plan, hardship program, and settlement alternatives
- **Reward:** a simulated combination of resolution probability, intervention cost, customer satisfaction, and default risk
- **Learning objective:** balance exploitation of treatments that currently appear effective with exploration of less-tested treatments

The contextual-bandit framing is intentionally simple because this prototype models a mostly **single-step treatment decision**. A full reinforcement-learning formulation would be more appropriate if the project modeled repeated customer states and treatment decisions over time.

### 3. Application Layer

A FastAPI layer demonstrates how model scoring and treatment recommendations could be exposed to another application.

## Simplified Architecture

```text
Synthetic customer data
        |
        v
XGBoost default-risk model
        |
        v
Customer context + risk score
        |
        v
LinUCB contextual bandit
        |
        v
Candidate hardship treatment
```

## Why a Contextual Bandit?

A classification model answers a prediction question such as:

> How likely is this customer to default?

A contextual bandit addresses a different decision question:

> Given what we know about this customer, which available treatment should we consider?

LinUCB estimates the expected reward of each action for the current context and adds an uncertainty bonus that encourages controlled exploration.

Conceptually:

```text
score(action) = predicted reward + exploration bonus
```

This provides a simple way to demonstrate the exploration-versus-exploitation trade-off.

## Project Structure

```text
resiliency/
├── data/
│   └── generator.py          # synthetic customer data
├── models/
│   ├── classifier.py         # XGBoost default-risk model
│   ├── linucb.py             # contextual-bandit implementation
│   └── rl_agent.py           # earlier Q-learning experiment
├── evaluation/
│   ├── metrics.py            # classification metrics
│   ├── ips.py                # experimental OPE utilities
│   └── ope.py                # experimental OPE utilities
└── utils/
    └── preprocessing.py

api/
├── main.py
└── schemas.py

scripts/
├── train.py                  # classifier + earlier RL experiment
├── train_bandit.py           # LinUCB experiment
└── generate_data.py

tests/
notebooks/
```

The Q-learning and off-policy-evaluation modules are retained as learning experiments, but **LinUCB is the primary treatment-recommendation approach presented in this project**.

## Quick Start

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Generate/train the base models

```bash
python scripts/train.py --n-samples 10000 --n-rl-episodes 10000
```

### Train the contextual bandit

```bash
python scripts/train_bandit.py
```

### Start the API

```bash
uvicorn api.main:app --reload --port 8080
```

### Run tests

```bash
pytest tests/ -v
```

## Treatment Actions

The LinUCB experiment uses a simplified action space:

| Treatment | Example use |
|---|---|
| Payment Plan | Moderate delinquency |
| Hardship Program | Recent or temporary hardship |
| Settlement 50% | Moderate-to-high default risk |
| Settlement 30% | More severe hardship / high default risk |

These descriptions and rewards are **simulated assumptions for demonstration purposes**.

## Limitations and Next Steps

This prototype is intentionally simplified.

A real-world implementation would require:

- historical treatment and outcome data rather than synthetic rewards;
- careful causal analysis to distinguish treatment effects from correlations;
- policy and regulatory constraints on available treatments;
- fairness and customer-outcome guardrails;
- reliable propensity logging and offline evaluation;
- controlled prospective experimentation before deployment;
- production monitoring for model and policy performance.

A more complete sequential RL formulation could also model repeated monthly decisions such as:

```text
current customer state
        -> treatment
        -> repayment / delinquency response
        -> updated customer state
        -> next treatment decision
```

That extension is outside the scope of this prototype.

## What I Learned

The main takeaway from this project is that **risk prediction and treatment optimization are different problems**. A strong predictive model can estimate customer risk, while a contextual bandit provides a framework for learning which action may be most appropriate for different customer contexts.

This repository is intended as a practical learning exercise connecting credit-risk domain knowledge with modern machine-learning decision methods.
