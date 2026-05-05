import os
import sys
import importlib
from typing import Any, Dict
from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv

# Ensure workers resolve the local soccer-twos package in this workspace.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_SOCCER_ENV_DIR = os.path.join(PROJECT_DIR, "soccer-twos-env")
if os.path.isdir(LOCAL_SOCCER_ENV_DIR) and LOCAL_SOCCER_ENV_DIR not in sys.path:
    sys.path.insert(0, LOCAL_SOCCER_ENV_DIR)

soccer_twos = importlib.import_module("soccer_twos")


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class PotentialRewardShapingWrapper(gym.core.Wrapper):
    """Adds dense, potential-based shaping rewards for single-agent training.

    This wrapper keeps the original sparse game reward and adds a bounded shaping
    term computed from potential differences between consecutive states.
    """

    def __init__(
        self,
        env,
        progress_weight: float = 1.0,
        distance_weight: float = 0.35,
        alignment_weight: float = 0.08,
        progress_scale: float = 20.0,
        distance_scale: float = 12.0,
        max_abs: float = 0.08,
        goal_direction: float = 1.0,
    ):
        super().__init__(env)
        self.progress_weight = float(progress_weight)
        self.distance_weight = float(distance_weight)
        self.alignment_weight = float(alignment_weight)
        self.progress_scale = max(float(progress_scale), 1e-6)
        self.distance_scale = max(float(distance_scale), 1e-6)
        self.max_abs = abs(float(max_abs))
        self.goal_direction = 1.0 if goal_direction >= 0 else -1.0
        self._prev_potential = None

    def reset(self, **kwargs):
        self._prev_potential = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extrinsic_reward = float(reward)

        shaping_reward = 0.0
        potential, components = self._compute_potential(info)
        if potential is not None and self._prev_potential is not None:
            shaping_reward = float(
                np.clip(potential - self._prev_potential, -self.max_abs, self.max_abs)
            )

        total_reward = extrinsic_reward + shaping_reward

        # Keep diagnostics in info so training scripts can inspect reward makeup.
        info_dict = dict(info) if isinstance(info, dict) else {}
        info_dict["reward_extrinsic"] = extrinsic_reward
        info_dict["reward_shaping"] = shaping_reward
        info_dict["reward_total"] = total_reward
        if components is not None:
            info_dict["reward_shaping_components"] = components

        self._prev_potential = None if done else potential
        return obs, total_reward, done, info_dict

    def _compute_potential(self, info: Any):
        if not isinstance(info, dict):
            return None, None

        player_info = info.get("player_info")
        ball_info = info.get("ball_info")
        if not player_info or not ball_info:
            return None, None

        try:
            player_pos = np.asarray(player_info["position"], dtype=np.float32)
            ball_pos = np.asarray(ball_info["position"], dtype=np.float32)
            ball_vel = np.asarray(ball_info["velocity"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None, None

        # Encourage moving the ball toward opponent goal (positive x for team 1).
        progress = self.goal_direction * float(ball_pos[0]) / self.progress_scale

        # Encourage controlled player to close down on the ball.
        distance = float(np.linalg.norm(player_pos - ball_pos))
        distance_term = -distance / self.distance_scale

        # Encourage ball velocity to point toward opponent goal.
        speed = float(np.linalg.norm(ball_vel))
        if speed > 1e-6:
            alignment = self.goal_direction * float(ball_vel[0]) / speed
        else:
            alignment = 0.0

        potential = (
            self.progress_weight * progress
            + self.distance_weight * distance_term
            + self.alignment_weight * alignment
        )

        components = {
            "progress": progress,
            "distance_term": distance_term,
            "alignment": alignment,
            "potential": potential,
        }
        return float(potential), components


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
            - worker_id_stride: spacing between Unity worker IDs to avoid port collisions.
            - reward_shaping: enable dense potential-based shaping for single-agent envs.
            - shaping_progress_weight: weight for ball progress term.
            - shaping_distance_weight: weight for player-to-ball distance term.
            - shaping_alignment_weight: weight for ball-velocity alignment term.
            - shaping_progress_scale: normalization for ball x-position.
            - shaping_distance_scale: normalization for player-ball distance.
            - shaping_max_abs: max absolute shaping reward added each step.
    """
    if hasattr(env_config, "worker_index"):
        worker_id_stride = int(env_config.get("worker_id_stride", 1))
        worker_slot = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
        env_config["worker_id"] = worker_slot * worker_id_stride
    env = soccer_twos.make(**env_config)

    if env_config.get("reward_shaping", False):
        # Team-vs-policy and other non-multiagent training expose scalar rewards,
        # so we can safely shape them here before passing to RLlib.
        env = PotentialRewardShapingWrapper(
            env,
            progress_weight=env_config.get("shaping_progress_weight", 1.0),
            distance_weight=env_config.get("shaping_distance_weight", 0.35),
            alignment_weight=env_config.get("shaping_alignment_weight", 0.08),
            progress_scale=env_config.get("shaping_progress_scale", 20.0),
            distance_scale=env_config.get("shaping_distance_scale", 12.0),
            max_abs=env_config.get("shaping_max_abs", 0.08),
            goal_direction=env_config.get("shaping_goal_direction", 1.0),
        )

    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
