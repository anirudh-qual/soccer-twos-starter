import os
import re
import sys
import yaml
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env, sample_pos_vel, sample_player, soccer_twos

EnvType = soccer_twos.EnvType
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_SOCCER_ENV_DIR = os.path.join(PROJECT_DIR, "soccer-twos-env")
if os.path.isdir(LOCAL_SOCCER_ENV_DIR) and LOCAL_SOCCER_ENV_DIR not in sys.path:
    sys.path.insert(0, LOCAL_SOCCER_ENV_DIR)


NUM_ENVS_PER_WORKER = 1
NUM_WORKERS = int(os.environ.get("SOCCER_NUM_WORKERS", "8"))
BASE_PORT = int(os.environ.get("SOCCER_BASE_PORT", "50039"))


def _checkpoint_iteration(path):
    match = re.search(r"checkpoint-(\d+)$", path)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(root):
    checkpoints = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("checkpoint-") and not filename.endswith(".tune_metadata"):
                checkpoints.append(os.path.join(dirpath, filename))
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_iteration)


SAFE_BASELINE_CHECKPOINT = (
    "ray_results/PPO_baseline_finetune/"
    "PPO_Soccer_56f40_00000_0_2026-04-24_10-10-25/"
    "checkpoint_001819/checkpoint-1819"
)
CURRICULUM_CHECKPOINT = (
    "ray_results/PPO_curriculum/"
    "PPO_Soccer_02186_00000_0_2026-04-23_23-59-36/"
    "checkpoint_001794/checkpoint-1794"
)
DEFAULT_RESTORE_CHECKPOINT = SAFE_BASELINE_CHECKPOINT
if os.environ.get("AUTO_RESTORE_LATEST", "0") == "1":
    DEFAULT_RESTORE_CHECKPOINT = (
        _latest_checkpoint(os.path.join("ray_results", "PPO_baseline_finetune"))
        or SAFE_BASELINE_CHECKPOINT
        if os.path.exists(SAFE_BASELINE_CHECKPOINT)
        else CURRICULUM_CHECKPOINT
    )
RESTORE_CHECKPOINT = os.environ.get("RESTORE_CHECKPOINT", DEFAULT_RESTORE_CHECKPOINT)
STOP_TIMESTEPS = int(os.environ.get("STOP_TIMESTEPS", "12000000"))
STOP_TIME_S = os.environ.get("STOP_TIME_S")

with open("curriculum.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]

BASELINE_FINE_TUNE = os.environ.get("BASELINE_FINE_TUNE", "1") != "0"

if BASELINE_FINE_TUNE:
    baseline_task = {
        "name": "Baseline Fine Tune",
        "config_fn": "ceia_baseline",
    }
    if os.environ.get("BASELINE_RANDOM_STARTS", "0") == "1":
        baseline_task["ranges"] = {
                "ball": {
                    "position": {"x": [-14, 14], "y": [-5, 5]},
                    "velocity": {"x": [-10, 10], "y": [-10, 10]},
                },
                "players": {
                    player: {
                        "rotation_y": [0, 360],
                        "position": {"x": [-14, 14], "y": [-5, 5]},
                        "velocity": {"x": [-10, 10], "y": [-10, 10]},
                    }
                    for player in range(4)
                },
        }
    tasks = [baseline_task]

DEFAULT_START_STAGE_NAME = "Baseline Fine Tune" if BASELINE_FINE_TUNE else "Hard Goal"
START_STAGE_NAME = os.environ.get("CURRICULUM_START_STAGE", DEFAULT_START_STAGE_NAME)
current = next(
    (idx for idx, task in enumerate(tasks) if task.get("name") == START_STAGE_NAME),
    0,
)


def _unwrap_team_vs_policy_env(env):
    """Returns the TeamVsPolicyWrapper even when nested in other gym wrappers."""
    current_env = env
    while current_env is not None:
        # Use strict type-name matching instead of hasattr() because gym.Wrapper
        # delegates attributes to inner envs and can create false positives.
        if current_env.__class__.__name__ == "TeamVsPolicyWrapper":
            return current_env
        current_env = getattr(current_env, "env", None)
    raise ValueError("Could not find TeamVsPolicyWrapper in environment stack.")


def _unwrap_shaping_env(env):
    """Returns the shaping wrapper when present, otherwise None."""
    current_env = env
    while current_env is not None:
        if hasattr(current_env, "max_abs") and hasattr(current_env, "_compute_potential"):
            return current_env
        current_env = getattr(current_env, "env", None)
    return None


def _resolve_train_policy(policies):
    if "default_policy" in policies:
        return policies["default_policy"]
    if "default" in policies:
        return policies["default"]
    return next(iter(policies.values()))


def _make_random_player_policy(team_vs_env):
    player_action_space = team_vs_env.env.action_space
    return lambda *_: player_action_space.sample()


class SelfPlayMixtureOpponent:
    """Opponent policy that mixes current-policy play with random actions."""

    def __init__(self, team_vs_env, policy, random_prob=0.25):
        self.team_vs_env = team_vs_env
        self.policy = policy
        self.random_prob = float(np.clip(random_prob, 0.0, 1.0))
        self.player_action_space = team_vs_env.env.action_space
        self.player_action_branches = len(self.player_action_space.nvec)
        self._cached_obs_id = None
        self._cached_actions = {
            2: self.player_action_space.sample(),
            3: self.player_action_space.sample(),
        }
        self._next_player = 2

    def _refresh_actions(self):
        last_obs = self.team_vs_env.last_obs
        if (
            last_obs is None
            or 2 not in last_obs
            or 3 not in last_obs
            or np.random.rand() < self.random_prob
        ):
            self._cached_actions[2] = self.player_action_space.sample()
            self._cached_actions[3] = self.player_action_space.sample()
            return

        team_obs = np.concatenate((last_obs[2], last_obs[3]))
        team_action, *_ = self.policy.compute_single_action(team_obs, explore=False)
        team_action = np.asarray(team_action, dtype=np.int64).reshape(-1)
        split = self.player_action_branches
        if team_action.size >= 2 * split:
            self._cached_actions[2] = team_action[:split]
            self._cached_actions[3] = team_action[split : 2 * split]
        else:
            self._cached_actions[2] = self.player_action_space.sample()
            self._cached_actions[3] = self.player_action_space.sample()

    def __call__(self, _obs):
        last_obs = self.team_vs_env.last_obs
        current_obs_id = id(last_obs)
        if current_obs_id != self._cached_obs_id:
            self._cached_obs_id = current_obs_id
            self._refresh_actions()
            self._next_player = 2

        player_id = self._next_player
        self._next_player = 3 if player_id == 2 else 2
        return self._cached_actions[player_id]


class BaselineOpponent:
    """Opponent policy adapter that controls players 2 and 3 with CEIA baseline."""

    def __init__(self, team_vs_env):
        from ceia_baseline_agent.agent_ray import RayAgent as BaselineAgent

        self.team_vs_env = team_vs_env
        self.player_action_space = team_vs_env.env.action_space
        self.baseline = BaselineAgent(team_vs_env.env)
        self._cached_obs_id = None
        self._cached_actions = {
            2: self.player_action_space.sample(),
            3: self.player_action_space.sample(),
        }
        self._next_player = 2

    def _refresh_actions(self):
        last_obs = self.team_vs_env.last_obs
        if last_obs is None or 2 not in last_obs or 3 not in last_obs:
            self._cached_actions[2] = self.player_action_space.sample()
            self._cached_actions[3] = self.player_action_space.sample()
            return

        try:
            actions = self.baseline.act({2: last_obs[2], 3: last_obs[3]})
        except Exception as exc:
            print(f"Baseline opponent action failed, sampling random action: {exc}")
            actions = {}

        self._cached_actions[2] = np.asarray(
            actions.get(2, self.player_action_space.sample()), dtype=np.int64
        )
        self._cached_actions[3] = np.asarray(
            actions.get(3, self.player_action_space.sample()), dtype=np.int64
        )

    def __call__(self, _obs):
        last_obs = self.team_vs_env.last_obs
        current_obs_id = id(last_obs)
        if current_obs_id != self._cached_obs_id:
            self._cached_obs_id = current_obs_id
            self._refresh_actions()
            self._next_player = 2

        player_id = self._next_player
        self._next_player = 3 if player_id == 2 else 2
        return self._cached_actions[player_id]


def _set_random_opponent(env, _policies, _task):
    team_vs_env = _unwrap_team_vs_policy_env(env)
    team_vs_env.set_opponent_policy(_make_random_player_policy(team_vs_env))


def _set_selfplay_mixed_opponent(env, policies, task):
    team_vs_env = _unwrap_team_vs_policy_env(env)
    policy = _resolve_train_policy(policies)
    random_prob = task.get("opponent_random_prob", 0.25)
    team_vs_env.set_opponent_policy(
        SelfPlayMixtureOpponent(
            team_vs_env=team_vs_env,
            policy=policy,
            random_prob=random_prob,
        )
    )


def _set_ceia_baseline_opponent(env, _policies, _task):
    team_vs_env = _unwrap_team_vs_policy_env(env)
    opponent = getattr(team_vs_env, "_ceia_baseline_opponent", None)
    if opponent is None:
        opponent = BaselineOpponent(team_vs_env)
        team_vs_env._ceia_baseline_opponent = opponent
    team_vs_env.set_opponent_policy(opponent)


def _apply_shaping_overrides(env, task):
    shaping_env = _unwrap_shaping_env(env)
    if shaping_env is None:
        return

    if "shaping_max_abs" in task:
        shaping_env.max_abs = float(task["shaping_max_abs"])
    if "shaping_progress_weight" in task:
        shaping_env.progress_weight = float(task["shaping_progress_weight"])
    if "shaping_distance_weight" in task:
        shaping_env.distance_weight = float(task["shaping_distance_weight"])
    if "shaping_alignment_weight" in task:
        shaping_env.alignment_weight = float(task["shaping_alignment_weight"])


config_fns = {
    "none": _set_random_opponent,
    "random_players": _set_random_opponent,
    "selfplay_mixed": _set_selfplay_mixed_opponent,
    "ceia_baseline": _set_ceia_baseline_opponent,
}


class CurriculumUpdateCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks

        episode.user_data["extrinsic_return"] = 0.0
        episode.user_data["saw_extrinsic_reward"] = False
        episode.user_data["shaping_return"] = 0.0
        task = tasks[current]

        for env in base_env.get_unwrapped():
            config_fn = task.get("config_fn", "none")
            config_fns[config_fn](env, policies, task)
            _apply_shaping_overrides(env, task)
            if "ranges" in task:
                env.env_channel.set_parameters(
                    ball_state=sample_pos_vel(task["ranges"]["ball"]),
                    players_states={
                        player: sample_player(task["ranges"]["players"][player])
                        for player in task["ranges"]["players"]
                    },
                )

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
        last_info = episode.last_info_for("agent0")
        if not isinstance(last_info, dict):
            return

        extrinsic = last_info.get("reward_extrinsic")
        shaping = last_info.get("reward_shaping")
        if extrinsic is not None:
            episode.user_data["extrinsic_return"] += float(extrinsic)
            episode.user_data["saw_extrinsic_reward"] = True
        if shaping is not None:
            episode.user_data["shaping_return"] += float(shaping)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        if episode.user_data.get("saw_extrinsic_reward", False):
            extrinsic_return = float(episode.user_data.get("extrinsic_return", 0.0))
        else:
            extrinsic_return = float(episode.total_reward)
        shaping_return = float(episode.user_data.get("shaping_return", 0.0))
        win = 1.0 if extrinsic_return > 0.0 else 0.0
        non_loss = 1.0 if extrinsic_return >= 0.0 else 0.0

        episode.custom_metrics["extrinsic_return"] = extrinsic_return
        episode.custom_metrics["shaping_return"] = shaping_return
        episode.custom_metrics["win"] = win
        episode.custom_metrics["non_loss"] = non_loss

    def on_train_result(self, **info):
        global current

        result = info["result"]
        task = tasks[current]
        custom_metrics = result.get("custom_metrics", {})
        win_rate = custom_metrics.get("win_mean")
        non_loss_rate = custom_metrics.get("non_loss_mean")
        extrinsic_mean = custom_metrics.get("extrinsic_return_mean")

        min_win_rate = task.get("advance_win_rate", 0.75)
        min_non_loss = task.get("advance_non_loss_rate", 0.8)
        min_extrinsic = task.get("advance_extrinsic_return", 1.0)

        ready_to_advance = (
            win_rate is not None
            and non_loss_rate is not None
            and extrinsic_mean is not None
            and win_rate >= min_win_rate
            and non_loss_rate >= min_non_loss
            and extrinsic_mean >= min_extrinsic
        )

        result["curriculum_stage"] = current
        result["curriculum_stage_name"] = task["name"]
        result["curriculum_win_rate"] = win_rate
        result["curriculum_non_loss_rate"] = non_loss_rate
        result["curriculum_extrinsic_return_mean"] = extrinsic_mean

        print(
            "Curriculum stats | "
            f"stage={current}:{task['name']} "
            f"win_rate={win_rate} non_loss_rate={non_loss_rate} "
            f"extrinsic_mean={extrinsic_mean}"
        )

        if ready_to_advance and current < len(tasks) - 1:
            print("---- Updating tasks!!! ----")
            current += 1
            print(f"Current task: {current} - {tasks[current]['name']}")


def _callbacks_factory():
    """Return callbacks instance for RLlib callback-factory config."""
    return CurriculumUpdateCallback()


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    stop_config = {"timesteps_total": STOP_TIMESTEPS}
    if STOP_TIME_S:
        stop_config["time_total_s"] = int(STOP_TIME_S)

    print(f"Restoring from: {RESTORE_CHECKPOINT}")
    print(f"Stopping with: {stop_config}")

    analysis = tune.run(
        "PPO",
        name="PPO_baseline_finetune" if BASELINE_FINE_TUNE else "PPO_curriculum",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": _callbacks_factory,
            "lr": float(os.environ.get("PPO_LR", "1e-6")),
            "entropy_coeff": float(os.environ.get("PPO_ENTROPY_COEFF", "0.0002")),
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                "worker_id_stride": 2,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "reward_shaping": os.environ.get("REWARD_SHAPING", "1") == "1",
                "shaping_progress_weight": float(os.environ.get("SHAPING_PROGRESS_WEIGHT", "0.15")),
                "shaping_distance_weight": float(os.environ.get("SHAPING_DISTANCE_WEIGHT", "0.04")),
                "shaping_alignment_weight": float(os.environ.get("SHAPING_ALIGNMENT_WEIGHT", "0.02")),
                "shaping_progress_scale": 20.0,
                "shaping_distance_scale": 12.0,
                "shaping_max_abs": float(os.environ.get("SHAPING_MAX_ABS", "0.0005")),
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "rollout_fragment_length": 3000,
            "batch_mode": "complete_episodes",
        },
        stop=stop_config,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
        # restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
    )

    # Prefer sparse objective for model selection; fallback to shaped mean.
    metric = "custom_metrics/extrinsic_return_mean"
    best_trial = analysis.get_best_trial(metric, mode="max")
    if best_trial is None:
        metric = "episode_reward_mean"
        best_trial = analysis.get_best_trial(metric, mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric=metric, mode="max"
    )
    print(best_checkpoint)
    print("Done training")
