# SimPol: Simulating Polarisation

SimPol investigates how the structure of political belief systems interacts with social influence to produce population-level political polarisation in 23 different countries. This is a work in progress, currently available as a [preprint on ArXiv](https://arxiv.org/pdf/2606.27968).

---

## Repository Structure

The repository is divided into two main components corresponding to the two stages of the project:

```
.
├── inference/         # Belief-network inference and analysis
├── abm/                # Agent-based modelling and polarisation analysis
├── environments/       # Conda environments for both components
├── README.md
└── .gitignore
```

---

## `inference/` — Belief-network inference

The `inference/` directory contains the belief-network inference component of the project.

It processes ESS survey data and infers country-level networks representing relationships between political beliefs. The directory contains:

* ESS data processing and preparation;
* correlation and partial-correlation network inference;
* country-level belief networks;
* network visualisation and comparison;
* analysis of network properties, including signed triangles;
* robustness analyses of the inferred networks.

The resulting country-level belief networks provide the empirical belief structures used by the agent-based model.

## `abm/` — Agent-based model

The `abm/` directory contains the agent-based modelling component of the project.

The model uses the inferred belief networks to simulate how individual belief systems evolve under internal coherence pressures and social influence. This component contains:

* the agent-based simulation implementation;
* simulation runners and parameter exploration;
* polarisation measurement;
* analysis of belief dynamics and polarisation trajectories;
* comparison of simulated polarisation across countries and model parameters.

Together, the two components form the main analysis pipeline:

```
ESS survey data
       ↓
Belief-network inference
       ↓
Country-level belief networks
       ↓
Agent-based model
       ↓
Emergent political polarisation
```

## Setup

The inference and ABM components use separate Conda environments.

Create the environments with:

```
conda env create -f environments/inference.yml
conda env create -f environments/abm.yml
```

Then activate the environment corresponding to the component you are working on:

`conda activate inference`

or:

`conda activate abm`

To update an existing environment after changes to its environment file:

`conda env update -f environments/inference.yml --prune`

or:

`conda env update -f environments/abm.yml --prune`