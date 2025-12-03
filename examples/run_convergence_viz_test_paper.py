import matplotlib
matplotlib.use("Agg")  # safe for headless / server env

import os

from map_llm_toolkit import (
    MAPModelRunner,
    project_pca,
    plot_convergence_trajectories,
)


def main():
    # ====== MODEL CONFIG (same pair as in the paper) ======
    models_to_run = [
        {
            "name": "Llama-3-8B",
            "path": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        },
        {
            "name": "Qwen-2.5-7B",
            "path": "Qwen/Qwen2.5-7B-Instruct",
        },
    ]

    # ====== PROMPTS (justice / fairness family, same semantics as paper) ======
    prompts = [
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

    trajectories_per_model = []

    # ====== RUN EACH MODEL VIA TOOLKIT API ======
    for cfg in models_to_run:
        model_name = cfg["path"]
        print(f"[MAP] Running convergence test for: {cfg['name']} ({model_name})")

        runner = MAPModelRunner(model_name)
        traj_hd = runner.get_layer_trajectories(prompts)  # list of (L, D)
        print(f"[MAP]  Collected {len(traj_hd)} trajectories, shape of first: {traj_hd[0].shape}")

        traj_2d = project_pca(traj_hd, n_components=2)    # list of (L, 2)
        print(f"[MAP]  Projected to 2D, shape of first: {traj_2d[0].shape}")

        runner.close()

        # This is exactly the structure expected by plot_convergence_trajectories:
        #   Sequence[(model_name, list_of_2D_trajectories)]
        trajectories_per_model.append((cfg["name"], traj_2d))

    # ====== VIZ: USE BUILT-IN FUNCTION ONLY ======
    fig = plot_convergence_trajectories(trajectories_per_model)

    out_path = "convergence_viz_test.png"
    fig.savefig(out_path, dpi=300)
    print(f"[MAP] Convergence visualization saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
