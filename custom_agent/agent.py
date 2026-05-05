import os
import pickle
from typing import Dict

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface

from .observation_wrapper import transform_observation


class RllibDQNDuelingNetwork(nn.Module):
    """RLlib TorchModelV2-compatible DQN dueling network with exact naming and structure."""
    def __init__(self, state_size: int, action_size: int, hidden_sizes=(256, 256)):
        super().__init__()
        # RLlib wraps each layer in nn.ModuleDict with key '_model'
        self._hidden_layers = nn.ModuleList([
            nn.ModuleDict({"_model": nn.Sequential(nn.Linear(state_size, hidden_sizes[0]), nn.ReLU())}),
            nn.ModuleDict({"_model": nn.Sequential(nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU())}),
        ])
        # Value branch (dueling)
        self.value_module = nn.ModuleDict({
            "dueling_V_0": nn.ModuleDict({"_model": nn.Sequential(nn.Linear(hidden_sizes[1], 1))}),
            "V": nn.Sequential(nn.Linear(hidden_sizes[1], 1)),
        })
        # Advantage branch (dueling)
        self.advantage_module = nn.ModuleDict({
            "dueling_A_0": nn.ModuleDict({"_model": nn.Sequential(nn.Linear(hidden_sizes[1], action_size))}),
            "A": nn.Sequential(nn.Linear(hidden_sizes[1], action_size)),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._hidden_layers:
            x = layer["_model"](x)
        value = self.value_module["dueling_V_0"]["_model"](x)
        advantage = self.advantage_module["dueling_A_0"]["_model"](x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class CustomAgent(AgentInterface):
    """DQN-based custom agent with transformed observations."""

    def __init__(self, env, checkpoint_path: str = None):
        super().__init__()
        self.flattener = ActionFlattener(env.action_space.nvec)
        self.action_size = self.flattener.action_space.n

        transformed_obs_dim = (int(env.observation_space.shape[0]) + 1) // 2 + 4
        self.model = self._build_model(transformed_obs_dim)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "checkpoint.pth"
            )

        try:
            if os.path.isdir(checkpoint_path):
                checkpoint_files = sorted(
                    f
                    for f in os.listdir(checkpoint_path)
                    if f.startswith("checkpoint-") and not f.endswith(".tune_metadata")
                )
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_path, checkpoint_files[-1])

            if os.path.isfile(checkpoint_path):
                if checkpoint_path.endswith(".pth"):
                    state = torch.load(checkpoint_path, map_location="cpu")
                    if isinstance(state, dict) and "model_state_dict" in state:
                        normalized = self._normalize_model_state(state["model_state_dict"])
                    else:
                        normalized = self._normalize_model_state(state)
                else:
                    with open(checkpoint_path, "rb") as f:
                        state = pickle.load(f)
                    worker_state = state.get("worker")
                    if isinstance(worker_state, bytes):
                        worker_state = pickle.loads(worker_state)

                    # Newer RLlib checkpoints.
                    if (
                        isinstance(worker_state, dict)
                        and "state" in worker_state
                        and "default_policy" in worker_state["state"]
                        and "policy_state" in worker_state["state"]["default_policy"]
                    ):
                        model_state = worker_state["state"]["default_policy"]["policy_state"]["model"]
                    # Older RLlib checkpoints may store policy weights directly.
                    elif (
                        isinstance(worker_state, dict)
                        and "state" in worker_state
                        and "default_policy" in worker_state["state"]
                    ):
                        model_state = worker_state["state"]["default_policy"]
                    else:
                        raise ValueError("Unsupported RLlib checkpoint structure")

                    normalized = self._normalize_model_state(model_state)

                expected_in = normalized["_hidden_layers.0._model.0.weight"].shape[1]
                current_in = self.model._hidden_layers[0]["_model"][0].in_features
                if expected_in != current_in:
                    self.model = self._build_model(expected_in)

                self.model.load_state_dict(normalized)
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}\nUsing random weights.")

        self.model.eval()

    def _build_model(self, state_size: int) -> RllibDQNDuelingNetwork:
        return RllibDQNDuelingNetwork(
            state_size=state_size,
            action_size=self.action_size,
            hidden_sizes=(256, 256),
        )

    @staticmethod
    def _normalize_model_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Accept both converted .pth state_dict and raw RLlib policy state formats.
        """
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dict")

        # RLlib policy state often nests actual model tensors under "weights".
        if "weights" in state and isinstance(state["weights"], dict):
            state = state["weights"]

        normalized = {}
        for key, value in state.items():
            if not hasattr(value, "shape"):
                continue
            if key.startswith("_value_branch."):
                continue

            new_key = key
            new_key = new_key.replace("advantage_module.A._model.0.", "advantage_module.A.0.")
            new_key = new_key.replace("value_module.V._model.0.", "value_module.V.0.")
            # RLlib checkpoints may store numpy arrays; PyTorch expects tensors.
            normalized[new_key] = torch.as_tensor(value)

        if not normalized:
            raise ValueError("No model tensor weights found in checkpoint state")
        return normalized

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions: Dict[int, np.ndarray] = {}

        for player_id, player_obs in observation.items():
            transformed = transform_observation(player_obs)
            state = torch.from_numpy(transformed).float().unsqueeze(0)

            with torch.no_grad():
                q_values = self.model(state)

            action_index = int(np.argmax(q_values.cpu().numpy()))
            actions[player_id] = self.flattener.lookup_action(action_index)

        return actions
