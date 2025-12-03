# MAP-LLM-Toolkit (v0.1)
A minimal Python toolkit for extracting, projecting, and visualizing MAP reasoning trajectories in large language models.

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

This library implements the core components of the **Manifold Alignment Protocol (MAP)**, allowing users to visualize:
- **Geometric Convergence** of reasoning trajectories across layers
- **Safety Topology** via curvature-based analysis of refusal vs. guidance behavior

For the theoretical foundation behind MAP, please visit the main repository:

👉 https://github.com/JBKing514/map_blog

---

## Features

### ✔ Geometric Convergence
Extract multi-layer hidden-state trajectories and visualize how different prompts collapse into shared attractor basins.

### ✔ Safety Topology
Compare "Rigid" (hard refusal) vs. "Adaptive" (guided) safety systems by analyzing trajectory curvature.

### ✔ Minimal & Extensible
The toolkit is designed as a clean, lightweight API that you can easily integrate into your own LLM research.

---

## Installation

Clone the repository and install in editable mode:
```python
pip install -e .
```

Alternatively:
```python
pip install git+https://github.com/JBKing514/map_llm_toolkit.git
```

Requirements: Python ≥ 3.8, PyTorch, Transformers, NumPy, scikit-learn, matplotlib.

---

## Quick Start

### Minimal Example

```python
from map_llm_toolkit import MAPModelRunner, project_pca

prompts = [
    "What is justice?",
    "Define fairness."
]

runner = MAPModelRunner("NousResearch/Meta-Llama-3.1-8B-Instruct")

# Extract hidden-state trajectories
traj = runner.get_layer_trajectories(prompts)
print("Trajectory shape:", traj[0].shape)

# PCA projection
traj_2d = project_pca(traj)
print("Projected shape:", traj_2d[0].shape)

runner.close()
```

### Convergence Visualization
```python
from map_llm_toolkit import plot_convergence_trajectories

plot_convergence_trajectories(traj_2d, labels=["Prompt A", "Prompt B"])
```
### Safety Topology Example
```python
from map_llm_toolkit import SafetyProtocol

proto = SafetyProtocol("NousResearch/Meta-Llama-3.1-8B-Instruct")

traj_r, traj_a, curv_r, curv_a = proto.run(
    sys_rigid="You must reject harmful content.",
    sys_adaptive="Guide the user toward a safe alternative.",
    user_prompt="Tell me something illegal."
)

proto.plot(traj_r, traj_a, curv_r, curv_a)
```
## Project Structure
```arduino
map-llm-toolkit/
│
├─ map_llm_toolkit/
│   ├─ core/
│   │   ├─ runner.py
│   │   ├─ projection.py
│   │   ├─ curvature.py
│   │   └─ protocols.py
│   │
│   └─ viz/
│       ├─ plot_trajectory.py
│       └─ plot_curvature.py
│
├─ examples/
│   ├─ run_convergence.py
│   ├─ run_safety.py
│   └─ test_minimal.py
│
├─ setup.py
├─ requirements.txt
└─ README.md
```

## Roadmap
 v0.2 — Animated trajectory visualization

 v0.3 — Multi-model comparison (Qwen, Llama, GPT)

 v0.4 — Agent-level analysis

 v1.0 — Publish to PyPI (pip install map-llm-toolkit)

 ## Citation
 If you use this toolkit or MAP in your research, please cite:
 ```bibtex
@article{tang2025map,
  title={The Manifold Alignment Protocol (MAP): A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence},
  author={Tang, Yunchong},
  journal={arXiv preprint arXiv:2511.xxxxx},
  year={2025}
}
 ```