import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
DEFAULT_RESTORE_CHECKPOINT = (
    "ray_results/PPO_selfplay_rec/"
    "PPO_Soccer_curr_agent_00000_0_imported_2026-04-24/"
    "checkpoint_002300/checkpoint-2300"
)
RESTORE_CHECKPOINT = os.environ.get("RESTORE_CHECKPOINT", DEFAULT_RESTORE_CHECKPOINT)
if RESTORE_CHECKPOINT and not os.path.exists(RESTORE_CHECKPOINT):
    RESTORE_CHECKPOINT = None


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"  # Choose 01 policy for agent_01
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.25, 0.125, 0.125],
        )[0]


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when reward is high enough
        """
        if info["result"]["episode_reward_mean"] > 0.5:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


if __name__ == "__main__":
    ray.init()
    if RESTORE_CHECKPOINT:
        print(f"Restoring PPO self-play training from: {RESTORE_CHECKPOINT}")
    else:
        print("No restore checkpoint found; starting PPO self-play training from scratch.")

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER,},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 7200,},  # 2h
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
