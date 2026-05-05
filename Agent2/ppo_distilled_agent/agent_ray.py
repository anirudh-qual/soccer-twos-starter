import pickle
import os
from typing import Dict

# Ray reads several env vars at import / ray.init; set before `import ray` so
# dashboard/metrics agents do not bind to an unresolvable hostname on HPC nodes.
for _key, _val in (
    ("RAY_DISABLE_DASHBOARD", "1"),
    ("RAY_DISABLE_IMPORT_WARNING", "1"),
    ("RAY_NODE_IP_ADDRESS", "127.0.0.1"),
):
    os.environ.setdefault(_key, _val)
os.environ.setdefault("HOSTNAME", "localhost")

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results",
    "PPO_distill_ceia",
    "checkpoint_final_15006048",
    "checkpoint_002450",
    "checkpoint-2450",
)
POLICY_NAME = "default"


class RayAgent(AgentInterface):
    """
    RayAgent loads an RLlib PPO policy checkpoint and runs inference.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPO distilled"

        # Always local (single-process): avoids raylets and the dashboard/metrics agent
        # on HPC nodes where those processes fail DNS / bind and spam or stall logs.
        print("[RayAgent] ray.init (local_mode, no dashboard)...", flush=True)
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
        )
        print("[RayAgent] ray.init done.", flush=True)

        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError("Could not find params.pkl in either the checkpoint dir or its parent directory!")

        config["num_workers"] = 0
        config["num_gpus"] = 0

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        print("[RayAgent] building trainer + restore (can take a minute)...", flush=True)
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)
        print("[RayAgent] ready.", flush=True)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(observation[player_id])
        return actions

