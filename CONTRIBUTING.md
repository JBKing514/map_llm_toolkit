# Contributing to MAP-LLM-Toolkit

Thank you for your interest in contributing!  
This project aims to provide a clean, extensible toolkit for visualizing geometric reasoning dynamics in LLMs.  
We welcome improvements, bug fixes, examples, and new visualization modules.

---

## How to Contribute

### 1. Fork the Repository
Create your own fork and clone it locally:

```python
git clone https://github.com/JBKing514/map_llm_toolkit.git
```
---

### 2. Create a Development Environment

```python
python -m venv env
source env/bin/activate # Windows: env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

### 3. Add Your Changes

Follow the existing file structure:
```python
map_llm_toolkit/
core/ # Algorithms, trajectory extraction, projections, curvature
viz/ # Plotting utilities
examples/ # Reproducible demo scripts
```

Please:

- Keep functions small and well-documented  
- Avoid breaking existing APIs  
- Add examples when adding new features  

---

## Coding Style Guidelines

- Use Python 3.8+ features where appropriate  
- Follow PEP8 unless there's a strong reason not to  
- Prefer explicit imports over wildcard imports  
- Write docstrings for public functions and classes  
- Use type hints whenever possible  

Example:

```python
def compute_curvature(points: np.ndarray) -> np.ndarray:
    """
    Compute discrete curvature for a sequence of 2D points.
    """
    ...
```
## Submitting a Pull Request (PR)

### Before submitting:

1. Ensure your code runs without errors

2. Add/Update examples if your feature introduces new functionality

3. Keep commits clean and descriptive

### Then open a PR with:

- A clear title

- A short description of the change

- Screenshots or example outputs (if applicable)

We review PRs as quickly as possible.

## Reporting Issues

If you find a bug or have a feature request:

1. Open a GitHub Issue

2. Include steps to reproduce

3. Attach logs, screenshots, or error messages

This helps us diagnose and fix problems faster.

## Roadmap Contributions

We welcome help on the following planned features:

- Animated trajectory visualization

- Multi-model comparison module

- Interactive dashboards (Plotly / Web UI)

- PyPI packaging

- Benchmark datasets for LLM geometry

If you'd like to take ownership of one of these, open an Issue titled:

Proposal: Implement <Feature Name>

And we will coordinate with you.

Thank You

Your contributions help advance MAP research and improve the toolkit for the entire community.
We deeply appreciate your support and creativity.

— Yunchong Tang

---
