# SimPol - Complexity72H 2026

Short description of the project.

## Repository Structure

```
.
├── abm/
│   └── __init__.py
├── inference/
│   └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

- `abm/` — agent-based modeling code.
- `inference/` — inference, prediction, or model-serving code.

## Requirements

- Python 3.11+ (or your preferred version)
- `venv` (included with Python)

## Creating a Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, upgrade pip:

```bash
pip install --upgrade pip
```

## Installing Dependencies

If a `requirements.txt` file exists:

```bash
pip install -r requirements.txt
```

To create or update dependencies:

```bash
pip freeze > requirements.txt
```

## Running Code

Example:

```bash
python inference/inference.py
```

or

```bash
python abm/abm.py
```

## Recommended Development Workflow

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements/inference.txt
```
or
```bash
pip install -r requirements/abm.txt
```

When adding a new dependency:

```bash
pip install package-name
pip freeze > requirements/inference.txt # or pip freeze > requirements/abm.txt
```

Deactivate the environment when finished:

```bash
deactivate
```

## Git Ignore

The repository should ignore virtual environments, caches, and build artifacts:

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