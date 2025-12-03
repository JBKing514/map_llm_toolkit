import gc
from typing import List, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MAPModelRunner:
    """
    Thin wrapper around a HF causal LM that exposes MAP-style trajectory APIs.

    - load(): lazy-loads model/tokenizer with output_hidden_states=True
    - close(): frees GPU/CPU memory
    - get_layer_trajectories(): single forward pass per prompt, per-layer last-token states
    - generate_trajectory(): autoregressive rollout with hidden states at each step
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        self._tokenizer = None
        self._model = None

    # ------------- lifecycle -------------

    def load(self) -> None:
        if self._model is not None:
            return
        print(f"[MAP] Loading model: {self.model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

    def close(self) -> None:
        if self._model is None:
            return
        print(f"[MAP] Releasing model: {self.model_name}")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ------------- MAP primitives -------------

    def get_layer_trajectories(self, prompts: List[str]) -> List[np.ndarray]:
        """
        MAP convergence experiment:
        For each prompt, run a single forward pass and collect
        the last-token vector at every layer.

        Returns
        -------
        trajectories : list of (num_layers, hidden_dim) arrays
        """
        self.load()
        model, tokenizer = self._model, self._tokenizer

        trajectories: List[np.ndarray] = []
        print(f"[MAP] Getting layer trajectories for {len(prompts)} prompts")

        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

            traj = []
            for layer_out in outputs.hidden_states:
                # (batch, seq, dim) -> last token of first example
                vec = layer_out[0, -1, :].detach().cpu().numpy().astype("float32")
                traj.append(vec)
            trajectories.append(np.stack(traj, axis=0))

        return trajectories

    def generate_trajectory(
        self,
        system_prompt: str,
        user_prompt: str,
        num_steps: int = 20,
        generation_kwargs: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        MAP safety experiment:
        Autoregressively generate tokens and record the last-layer
        last-token hidden state at each step.

        Parameters
        ----------
        system_prompt : str
        user_prompt   : str
        num_steps     : int
            Number of generation steps to observe.
        generation_kwargs : dict
            Passed through to model (e.g., temperature, top_p).

        Returns
        -------
        traj : (num_steps, hidden_dim) array
        """
        self.load()
        model, tokenizer = self._model, self._tokenizer
        generation_kwargs = generation_kwargs or {}

        text = system_prompt + "\n\nUser: " + user_prompt + "\n\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        current_ids = inputs.input_ids
        trajectory = []

        for _ in range(num_steps):
            with torch.no_grad():
                outputs = model(current_ids, output_hidden_states=True, **generation_kwargs)

            last_hidden = outputs.hidden_states[-1][0, -1, :].detach().cpu().numpy()
            trajectory.append(last_hidden)

            # greedy next token
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return np.stack(trajectory, axis=0)
