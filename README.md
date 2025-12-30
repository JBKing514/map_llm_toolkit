# MAP-LLM-Toolkit (v0.1)
A minimal Python toolkit for extracting, projecting, and visualizing MAP reasoning trajectories in large language models.

[![Standard Version](https://img.shields.io/badge/MAP%20Spec-v1.0.0-orange)](https://doi.org/10.5281/zenodo.18091447) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17900444.svg)](https://doi.org/10.5281/zenodo.17900444) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE) 

This library implements the core components of the **Manifold Alignment Protocol (MAP)**, allowing users to visualize:
- **Geometric Convergence** of reasoning trajectories across layers
- **Safety Topology** via curvature-based analysis of refusal vs. guidance behavior
- **Semantic Alignment** through layer-wise alignment metrics (A and ΔA) comparing tight vs. sparse semantic prompt clusters

For the theoretical foundation behind MAP, please visit the main repository:

👉 https://github.com/JBKing514/map_blog

**Note:**
This project installs the CPU version of PyTorch by default.
If you want to run the experiments on GPU, please manually install a CUDA-enabled build of PyTorch that matches your CUDA version (e.g., cu118 or cu121).
After installing the appropriate GPU build, all scripts will automatically use the GPU if available.

---

## Features

### ✔ Geometric Convergence
Extract multi-layer hidden-state trajectories and visualize how different prompts collapse into shared attractor basins.

### ✔ Safety Topology
Compare "Rigid" (hard refusal) vs. "Adaptive" (guided) safety systems by analyzing trajectory curvature.

### ✔ Semantic Alignment (A & ΔA)
Quantify layer-wise semantic coherence across prompts by computing alignment profiles for tight vs. sparse prompt clusters, and visualize semantic separation (ΔA) between them.

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

The example script can be found in the `examples` folder with some pre-rendered JPGs for confirmation.

Alternatively, you can follow the minimal example below.

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

## Roadmap
 v0.2 — Animated trajectory visualization

 v0.3 — Multi-model comparison (Qwen, Llama, GPT)

 v0.4 — Agent-level analysis

 v1.0 — Publish to PyPI (pip install map-llm-toolkit)

## Contribute

If you want to contribute this project, please read our [Contribution/Issue Guideline](https://github.com/JBKing514/map_llm_toolkit/blob/main/CONTRIBUTING.md).

 ## Citation
 If you use MAP or its toolkits in your research, please cite:
 ```bibtex
@article{tang2025map,
  title={Manifold Alignment Protocol (MAP) Specification},
  author={Tang, Yunchong},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18091447},
  url={[https://doi.org/10.5281/zenodo.18091447](https://doi.org/10.5281/zenodo.18091447)}
}
 ```


## License

MIT License © Yunchong Tang
