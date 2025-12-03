from map_llm_toolkit import MAPModelRunner, project_pca

# Set minimal prompts
prompts = [
    "What is justice?",
    "Define fairness.",
]

# Load model
runner = MAPModelRunner("NousResearch/Meta-Llama-3.1-8B-Instruct")

# 1. Get multi-layer hidden state trajectories
traj = runner.get_layer_trajectories(prompts)

print("Number of trajectories:", len(traj))
print("One trajectory shape:", traj[0].shape)   # (num_layers, dim)

# 2. PCA projection
traj_2d = project_pca(traj)

print("Projected 2D trajectory shape:", traj_2d[0].shape)

runner.close()

print("✓ Minimal MAP test finished successfully!")
