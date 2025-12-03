from map_llm_toolkit import MAPModelRunner, project_pca, plot_convergence_trajectories

PROMPTS = [
    "Define the concept of justice.",
    "What does it mean to be fair?",
    "Explain the essence of legal equity.",
    "Describe the philosophical basis of justice.",
    "In simple terms, what is justice?",
    "Elaborate on the definition of fairness in society.",
    "What is the core principle of a just legal system?",
    "How would you define true equity?",
    "Summarize the idea of justice.",
    "Give a definition for the word justice.",
]

MODELS = [
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def main():
    results = []
    for name in MODELS:
        runner = MAPModelRunner(name)
        traj = runner.get_layer_trajectories(PROMPTS)
        traj_2d = project_pca(traj, n_components=2)
        runner.close()
        results.append((name, traj_2d))

    fig = plot_convergence_trajectories(results)
    fig.savefig("convergence_dual_model.png", dpi=300)


if __name__ == "__main__":
    main()
