# Project Name

Short description of the repository.

---

# Repository Structure

```text
.
├── abm/
│   ├── __init__.py
│   └── ...
├── inference/
│   ├── __init__.py
│   └── ...
├── requirements/
│   ├── abm.txt
│   ├── inference.txt
│   └── dev.txt
├── README.md
└── .gitignore
```

This repository contains two largely independent codebases:

* **Inference Team** (`inference/`)
* **ABM Team** (`abm/`)

Each team maintains its own dependencies while sharing the same repository.

---

# Python Environment Setup

Create a virtual environment at the repository root:

```bash
python -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

# ABM Team

The ABM team develops and maintains the code in:

```text
abm/
```

## Install ABM Dependencies

```bash
pip install -r requirements/abm.txt
```

## Run ABM Code

Example:

```bash
python abm/abm.py
```

## Updating Dependencies

After adding or upgrading packages:

```bash
pip freeze > requirements/abm.txt
```

Only include dependencies required by the ABM codebase.

---

# Inference Team

The Inference team develops and maintains the code in:

```text
inference/
```

## Install Inference Dependencies

```bash
pip install -r requirements/inference.txt
```

## Run Inference Code

Example:

```bash
python inference/inference.py
```

## Updating Dependencies

After adding or upgrading packages:

```bash
pip freeze > requirements/inference.txt
```

Only include dependencies required by the inference codebase.

---

# Working Across Both Teams

If you need both environments:

```bash
pip install -r requirements/abm.txt
pip install -r requirements/inference.txt
pip install -r requirements/dev.txt
```

---

# Deactivating the Environment

When finished:

```bash
deactivate
```

---

# .gitignore

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.coverage
dist/
build/
*.egg-info/
.idea/
.vscode/
.DS_Store
```
