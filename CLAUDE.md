# SimPol — Claude Code Guide

## Project Overview

**SimPol** infers a belief network from the European Social Survey (ESS) and runs an agent-based model (ABM) on that network to study polarization — when it arises, under what conditions, and how it varies by country or sociodemographic group.

Two independent workstreams share this repo:

| Workstream | Directory | Conda env |
|---|---|---|
| Belief-network inference | `inference/` | `complexity72-simpol` (`environments/inference.yml`) |
| Agent-based model | `abm/` | `abm` (`environments/abm.yml`) |

## Key Commands

```bash
# Activate the shared inference env (graph-tool, networkx, seaborn)
conda activate complexity72-simpol

# Run inference pipeline
python inference/main.py

# Run ABM
python abm/main.py

# Export env after adding packages
conda env export --from-history > environments/inference.yml
conda env export --from-history > environments/abm.yml
```

## Architecture

```
C72H-SimPol/
├── inference/        # ESS data loading, belief-network construction, BN inference
│   ├── __init__.py
│   └── inference.py
├── abm/              # Agent-based model — opinion dynamics on the belief network
│   ├── __init__.py
│   └── abm.py
├── environments/
│   ├── inference.yml
│   └── abm.yml
└── CLAUDE.md
```

## Domain Context

- **Data**: European Social Survey (ESS) — multi-country, multi-wave attitudinal survey data.
- **Belief network inference**: Learn a graphical model (Bayesian network or similar) over ESS attitude variables, possibly stratified by country or sociodemographic group.
- **ABM**: Agents hold belief states consistent with the inferred network. Dynamics explore how local interaction rules lead to macroscopic polarization patterns.
- **Key research questions**: Under what conditions does polarization emerge? Do different countries or demographic groups polarize differently?

## Coding Conventions

- Python 3.12+, numpy/pandas for data, networkx / graph-tool for graph operations, matplotlib/seaborn for viz.
- Each workstream (`inference/`, `abm/`) is self-contained — don't import across workstreams.
- No shared state between the two Conda environments.
- Prefer scripts (`main.py`) as entry points; library code lives in the module files.

## Running Tests

No test suite yet. When adding one, prefer pytest under each workstream directory.

## Claude Tips for This Project

- When working on inference, think in terms of conditional independence, Markov blankets, and score-based or constraint-based structure learning.
- When working on the ABM, think about agent update rules (bounded confidence, Deffuant, etc.), network topology effects, and macro-level polarization metrics (bimodality, variance, group distance).
- Polarization analysis often requires stratified runs — by country (ESS `cntry` variable) or by demographic (age cohort, education level, left-right self-placement).
