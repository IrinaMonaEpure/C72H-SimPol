# Project Name

Short description of the project.

---

## Repository Structure

```text
.
├── abm/
│   └── ...
├── inference/
│   └── ...
├── environments/
│   ├── abm.yml
│   ├── inference.yml
├── README.md
└── .gitignore
```

This repository contains two independent workstreams:

* **ABM Team** (`abm/`)
* **Inference Team** (`inference/`)

Each team maintains its own Conda environment.

---

# Prerequisites

Install either:

* Miniconda
* Anaconda

---

# Inference Team

Code owned by the Inference team lives in:

```text
inference/
```

## Create the Environment

```bash
conda env create -f environments/inference.yml
```

## Activate the Environment

```bash
conda activate inference
```

## Run Inference Code

```bash
python inference/main.py
```

## Update Dependencies

After installing new packages:

```bash
conda env export --from-history > environments/inference.yml
```

Only include packages required by the inference codebase.

---

# ABM Team

Code owned by the ABM team lives in:

```text
abm/
```

## Create the Environment

```bash
conda env create -f environments/abm.yml
```

## Activate the Environment

```bash
conda activate abm
```

## Run ABM Code

```bash
python abm/main.py
```

## Update Dependencies

After installing new packages:

```bash
conda env export --from-history > environments/abm.yml
```

Only include packages required by the ABM codebase.

---

# Working Across Teams

If you contribute to both teams, switch environments as needed:

```bash
conda activate abm
```

or

```bash
conda activate inference
```

Each environment remains isolated, preventing dependency conflicts between the ABM and inference codebases.

---

# Updating an Existing Environment

After modifying an environment file:

```bash
conda env update -f environments/abm.yml --prune
```

or

```bash
conda env update -f environments/inference.yml --prune
```

---

# Deactivate the Environment

```bash
conda deactivate
```

---

# Example Environment File

`environments/abm.yml`

```yaml
name: abm
channels:
  - conda-forge

dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
```

`environments/inference.yml`

```yaml
name: inference
channels:
  - conda-forge

dependencies:
  - python=3.12
  - pytorch
  - transformers
  - pip
```

---

# .gitignore

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/

.env
.venv/

.idea/
.vscode/

build/
dist/
*.egg-info/

.DS_Store
```
