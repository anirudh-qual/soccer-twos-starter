import numpy as np
import gym
from gym import spaces
from ray.rllib import MultiAgentEnv


class RewardShapingWrapper(gym.Wrapper, MultiAgentEnv):
    """
    Environment wrapper that applies reward shaping to accelerate learning.

    Modifications:
    1. Ball Proximity Reward: +0.01 reward for being closer to the ball than last step
    2. Goal Direction Reward: +0.01 reward for moving towards opponent goal
    3. Action Magnitude Penalty: -0.001 * action_magnitude to encourage efficient movement

    These shaped rewards provide intermediate feedback in the sparse reward environment,
    helping agents learn basic skills (moving toward ball, positioning) before learning
    to score goals.
    """

    def __init__(self, env):
        """Initialize the wrapper"""
        super().__init__(env)
        self.last_distances_to_ball = {}
        self.last_goal_direction = {}

    def _get_player_state(self, state_vector, agent_id):
        """
        Extract player state from observation vector.
        For soccer_twos, observations include: [rays, position, velocity, ball_info, ...]
        Position is typically at indices ~85-87 for agent position
        """
        # Extract position (approximate indices, check actual env for exact values)
        # This is a general extraction that works with most soccer_twos configurations
        if len(state_vector) > 50:
            # Typical soccer_twos observation has 200+ dims
            # Position info is around index 85-87
            agent_x = state_vector[85] if len(state_vector) > 85 else 0
            agent_y = state_vector[86] if len(state_vector) > 86 else 0
            # Ball info is typically around index 95-97
            ball_x = state_vector[95] if len(state_vector) > 95 else 0
            ball_y = state_vector[96] if len(state_vector) > 96 else 0
        else:
            agent_x = agent_y = ball_x = ball_y = 0

        return np.array([agent_x, agent_y]), np.array([ball_x, ball_y])

    def _calculate_shaped_reward(self, agent_id, obs, action, base_reward):
        """
        Calculate additional shaped reward components.

        Args:
            agent_id: ID of the agent
            obs: Observation vector
            action: Action taken (continuous)
            base_reward: Original environment reward

        Returns:
            Total reward with shaping applied
        """
        shaped_reward = base_reward
        penalty = 0.0

        # Extract agent and ball positions
        agent_pos, ball_pos = self._get_player_state(obs, agent_id)

        # 1. Ball Proximity Reward: reward getting closer to ball
        current_distance_to_ball = np.linalg.norm(agent_pos - ball_pos)

        if agent_id in self.last_distances_to_ball:
            distance_delta = (
                self.last_distances_to_ball[agent_id] - current_distance_to_ball
            )
            if distance_delta > 0:  # Getting closer
                shaped_reward += 0.01 * distance_delta

        self.last_distances_to_ball[agent_id] = current_distance_to_ball

        # 2. Goal Direction Reward: reward moving towards opponent goal
        # Opponent goal is at (x_max, 0) approximately
        opponent_goal_pos = np.array([9.0, 0.0])  # Typical arena boundary
        distance_to_goal = np.linalg.norm(agent_pos - opponent_goal_pos)

        if agent_id in self.last_goal_direction:
            goal_distance_delta = (
                self.last_goal_direction[agent_id] - distance_to_goal
            )
            if goal_distance_delta > 0:  # Getting closer to goal
                shaped_reward += 0.01 * goal_distance_delta

        self.last_goal_direction[agent_id] = distance_to_goal

        # 3. Action Magnitude Penalty: small penalty for large actions
        # This encourages efficient, smaller movements
        action_magnitude = np.linalg.norm(action) if hasattr(action, "__len__") else 0
        penalty = 0.001 * action_magnitude

        return shaped_reward - penalty

    def step(self, action):
        """
        Execute environment step with reward shaping.

        Args:
            action: Dictionary of actions for each agent

        Returns:
            obs, reward, done, info (all with reward shaping applied)
        """
        obs, base_reward, done, info = self.env.step(action)

        # Apply reward shaping for each agent
        shaped_reward = {}
        for agent_id in base_reward:
            shaped_reward[agent_id] = self._calculate_shaped_reward(
                agent_id, obs[agent_id], action.get(agent_id, 0), base_reward[agent_id]
            )

        return obs, shaped_reward, done, info

    def reset(self):
        """Reset environment and tracking variables"""
        self.last_distances_to_ball = {}
        self.last_goal_direction = {}
        return self.env.reset()
