"""
Train CustomAgent with DQN.

Usage:
    python train_custom_agent.py
    python train_custom_agent.py --restore <checkpoint_path>
"""

import argparse
import os

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1
TOTAL_TIMESTEPS = 10_000_000
DEFAULT_NUM_WORKERS = 12


def create_env_plain_dqn(env_config: dict = None):
    """Create a single-player env without observation wrapper changes."""
    if env_config is None:
        env_config = {}

    raw_env_config = env_config
    config = dict(env_config)
    config["variation"] = EnvType.team_vs_policy
    config["multiagent"] = False
    config["flatten_branched"] = True
    config["single_player"] = True
    # Ensure each worker gets a unique Unity worker/base port even when
    # rollout and evaluation workers have overlapping worker_index values.
    worker_index = getattr(raw_env_config, "worker_index", None)
    if worker_index is None:
        worker_index = config.get("worker_index", 1)
    vector_index = getattr(raw_env_config, "vector_index", None)
    if vector_index is None:
        vector_index = config.get("vector_index", 0)
    worker_index = int(worker_index)
    vector_index = int(vector_index)
    pid_component = os.getpid() % 10000
    unique_worker_id = pid_component + (worker_index * 10) + vector_index + 1
    config["worker_id"] = unique_worker_id
    config["base_port"] = 10000 + unique_worker_id * 2

    return create_rllib_env(config)


def main(args):
    ray.init()

    tune.registry.register_env("SoccerCustomDQN", create_env_plain_dqn)


    stop_timesteps = (
        args.stop_timesteps
        if hasattr(args, "stop_timesteps") and args.stop_timesteps is not None
        else TOTAL_TIMESTEPS
    )
    checkpoint_freq = (
        args.checkpoint_freq
        if hasattr(args, "checkpoint_freq") and args.checkpoint_freq is not None
        else 5
    )
    experiment_name = (
        args.experiment_name
        if hasattr(args, "experiment_name") and args.experiment_name is not None
        else "DQN_CustomAgent_PlainEnv_v1"
    )

    config = {
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "log_level": "WARN",
        "framework": "torch",
        "env": "SoccerCustomDQN",
        "env_config": {
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "dueling": args.dueling,      # Toggle dueling architecture from CLI.
        "noisy": True,                # Noisy Nets (moved to top-level)
        "num_atoms": 51,              # Distributional RL (C51, moved to top-level)
        "double_q": True,             # Double DQN
        "n_step": 3,                  # Multi-step returns
        "replay_buffer_config": {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 500_000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "replay_sequence_length": 1,
            "learning_starts": 20_000,
        },
        "gamma": 0.99,
        "lr": 1e-4,
        "target_network_update_freq": 4000,
        "rollout_fragment_length": 16,
        "timesteps_per_iteration": 10_000,
        "train_batch_size": 4096,
        # Keep epsilon schedule explicit as a fallback even when NoisyNets is enabled.
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 1_000_000,
        },
        # Keep training stable first; enable evaluation in a separate run if needed.
        "evaluation_interval": 0,
        # Avoid infinite episodes when env does not provide max_episode_steps.
        "horizon": 1000,
        "batch_mode": "truncate_episodes",
    }

    analysis = tune.run(
        "DQN",
        name=experiment_name,
        config=config,
        stop={"timesteps_total": stop_timesteps},
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=args.restore,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )

    print(f"Best trial: {best_trial}")
    print(f"Best checkpoint: {best_checkpoint}")
    if best_trial is not None:
        print(f"Best reward mean: {best_trial.last_result.get('episode_reward_mean', 'N/A')}")

    print("Training complete.")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CustomAgent with DQN"
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=None,
        help="Number of timesteps to train (overrides default)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Checkpoint frequency (in training iterations)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for Ray Tune run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of rollout workers (default: 12)",
    )
    parser.add_argument(
        "--dueling",
        dest="dueling",
        action="store_true",
        default=True,
        help="Enable dueling network architecture (default: enabled)",
    )
    parser.add_argument(
        "--no-dueling",
        dest="dueling",
        action="store_false",
        help="Disable dueling network architecture",
    )

    args = parser.parse_args()
    main(args)
