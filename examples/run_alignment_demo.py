"""
Minimal alignment demo using MAP LLM Toolkit.

This script reproduces the layer-wise alignment A(ℓ) and ΔA(ℓ)
for tight vs. sparse semantics on two models:
  - NousResearch/Meta-Llama-3.1-8B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
"""

from map_llm_toolkit import (
    MAPModelRunner,
    compute_alignment_delta,
    plot_alignment_profiles,
)

# ---- Semantic prompt sets ----

TIGHT_PARAPHRASES = [
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

SPARSE_RANDOM = [
    "What will the weather be like in Tokyo tomorrow?",
    "Give me a simple recipe for chocolate cake.",
    "Explain the Pythagorean theorem in geometry.",
    "Who won the last World Cup in football?",
    "How does photosynthesis work in plants?",
    "Write a short story about a robot who learns to paint.",
    "What are the health benefits of regular exercise?",
    "Explain the concept of inflation in economics.",
    "Describe the main causes of climate change.",
    "How can I improve my time management skills?",
]


def run_for_model(model_name: str, hub_path: str):
    print(f"[MAP] Running alignment demo for: {model_name} ({hub_path})")
    runner = MAPModelRunner(hub_path)

    # 1. Extract layer trajectories for tight / sparse semantics
    tight_traj = runner.get_layer_trajectories(TIGHT_PARAPHRASES)
    sparse_traj = runner.get_layer_trajectories(SPARSE_RANDOM)

    # 2. Compute A_tight, A_sparse, ΔA
    L, A_tight, A_sparse, DeltaA = compute_alignment_delta(
        tight_traj,
        sparse_traj,
    )

    print(
        f"[MAP]  {model_name}: layers={len(L)}, "
        f"A_tight[0]={A_tight[0]:.3f}, A_sparse[0]={A_sparse[0]:.3f}, "
        f"ΔA_mean={DeltaA.mean():.3f}"
    )

    runner.close()

    return {
        "name": model_name,
        "L": L,
        "A_tight": A_tight,
        "A_sparse": A_sparse,
        "DeltaA": DeltaA,
    }


def main():
    results = []

    # Llama-3-8B
    results.append(
        run_for_model(
            model_name="Llama-3-8B",
            hub_path="NousResearch/Meta-Llama-3.1-8B-Instruct",
        )
    )

    # Qwen-2.5-7B
    results.append(
        run_for_model(
            model_name="Qwen-2.5-7B",
            hub_path="Qwen/Qwen2.5-7B-Instruct",
        )
    )

    # 3. Plot combined figure
    plot_alignment_profiles(
        results,
        save_path="alignment_delta_profile.png",
    )

    print("[MAP] ✓ Alignment demo finished successfully!")


if __name__ == "__main__":
    main()
