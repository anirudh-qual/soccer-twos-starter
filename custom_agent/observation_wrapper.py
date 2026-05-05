from typing import Dict, Union

import gym
import numpy as np
from gym import spaces


def transform_observation(obs: np.ndarray) -> np.ndarray:
    """
    Transform Soccer-Twos observation for DQN:
    - Use relative coordinates for ball and other players
    - Normalize all features to [-1, 1] (using field/velocity limits)
    - Add engineered features: distance to ball, angle to goal
    """
    obs = np.asarray(obs, dtype=np.float32)

    # --- Assumed observation layout (Soccer-Twos default) ---
    # [0:2]   agent pos (x, y)
    # [2]     agent rotation_y (degrees)
    # [3:5]   agent velocity (x, y)
    # [5:7]   ball pos (x, y)
    # [7:9]   ball velocity (x, y)
    # [9:]    other players' pos/vel (order varies by env)

    # Field/velocity limits (approximate, adjust as needed)
    FIELD_X = 10.0
    FIELD_Y = 7.0
    MAX_VEL = 6.0
    MAX_ROT = 180.0

    # Agent features
    agent_pos = obs[0:2]
    agent_rot = obs[2]
    agent_vel = obs[3:5]
    # Ball features
    ball_pos = obs[5:7]
    ball_vel = obs[7:9]

    # Relative ball position
    rel_ball_pos = (ball_pos - agent_pos) / np.array([FIELD_X, FIELD_Y])
    rel_ball_vel = ball_vel / MAX_VEL
    agent_vel_norm = agent_vel / MAX_VEL
    agent_rot_norm = agent_rot / MAX_ROT

    # Engineered features
    dist_to_ball = np.linalg.norm(ball_pos - agent_pos) / np.sqrt(FIELD_X**2 + FIELD_Y**2)
    goal_center = np.array([FIELD_X, 0.0])  # Assume right-side goal for agent 0
    vec_to_goal = goal_center - agent_pos
    angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0]) / np.pi  # [-1, 1]

    # Other players (positions and velocities, relative to agent)
    others = obs[9:]
    n_others = (len(others)) // 4
    rel_others = []
    for i in range(n_others):
        o_pos = others[i*4:i*4+2]
        o_vel = others[i*4+2:i*4+4]
        rel_pos = (o_pos - agent_pos) / np.array([FIELD_X, FIELD_Y])
        rel_vel = o_vel / MAX_VEL
        rel_others.extend([*rel_pos, *rel_vel])
    rel_others = np.array(rel_others, dtype=np.float32)

    # Concatenate all features
    features = np.concatenate([
        rel_ball_pos, rel_ball_vel, agent_vel_norm, [agent_rot_norm],
        rel_others, [dist_to_ball, angle_to_goal]
    ])

    # Normalize to [-1, 1] (already mostly normalized)
    features = np.clip(features, -1.0, 1.0)
    return features.astype(np.float32)


class ObservationTransformWrapper(gym.Wrapper):
    """Environment wrapper that applies the custom observation transform."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("ObservationTransformWrapper expects a Box observation space")

        if len(env.observation_space.shape) != 1:
            raise ValueError("ObservationTransformWrapper expects 1D observations")

        # Dynamically determine transformed observation shape
        sample_obs = env.observation_space.sample()
        transformed_sample = transform_observation(sample_obs)
        transformed_dim = transformed_sample.shape[0]

        # Set bounds to [-1, 1] (since transform_observation clips to this)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(transformed_dim,),
            dtype=np.float32,
        )

    def _transform_any(self, obs: Union[np.ndarray, Dict[int, np.ndarray]]):
        if isinstance(obs, dict):
            return {agent_id: transform_observation(agent_obs) for agent_id, agent_obs in obs.items()}
        return transform_observation(obs)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._transform_any(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._transform_any(obs), reward, done, info
