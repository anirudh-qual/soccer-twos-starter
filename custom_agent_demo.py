"""
Quick start guide for using CustomAgent.

This script demonstrates how to use the CustomAgent in different scenarios.
"""

import soccer_twos
from soccer_twos import EnvType
from custom_agent import CustomAgent


def demo_basic_usage():
    """Basic usage example"""
    print("=== CustomAgent Basic Usage ===\n")

    # Create environment
    env = soccer_twos.make(variation=EnvType.multiagent_team)

    # Initialize agent
    agent = CustomAgent(env)

    # Run a single episode
    obs = env.reset()
    total_reward = 0

    for step in range(100):
        # Get actions from agent
        actions = agent.act(obs)

        # Step environment
        obs, rewards, done, info = env.step(actions)

        # Sum rewards
        total_reward += sum(rewards.values())

        if done:
            break

    print(f"Episode completed in {step} steps")
    print(f"Total reward: {total_reward:.2f}")

    env.close()


def demo_vs_random():
    """Example: CustomAgent vs Random Agent (team vs team)"""
    print("\n=== CustomAgent vs Random Agents ===\n")

    env = soccer_twos.make(variation=EnvType.multiagent_team)
    agent = CustomAgent(env)

    obs = env.reset()
    custom_goals = 0
    random_goals = 0

    for step in range(500):
        # CustomAgent actions
        custom_actions = agent.act(obs)

        # Random actions (only for other team if env supports it)
        random_actions = {
            agent_id: env.action_space.sample()
            for agent_id in obs
            if agent_id not in custom_actions
        }
        all_actions = {**custom_actions, **random_actions}

        # Step
        obs, rewards, done, info = env.step(all_actions)

        # Track goals
        for agent_id, reward in rewards.items():
            if reward >= 1.0:
                if agent_id < 2:
                    custom_goals += 1
                else:
                    random_goals += 1

        if done:
            break

    print(f"Final score: CustomAgent {custom_goals} - {random_goals} Random")
    print(f"Winner: {'CustomAgent' if custom_goals > random_goals else 'Random' if random_goals > custom_goals else 'Draw'}")

    env.close()


def demo_load_custom_checkpoint(checkpoint_path):
    """Load agent with custom checkpoint"""
    print(f"\n=== Loading Custom Checkpoint ===\n")
    print(f"Checkpoint path: {checkpoint_path}")

    env = soccer_twos.make(variation=EnvType.multiagent_team)
    agent = CustomAgent(env, checkpoint_path=checkpoint_path)

    print("Agent loaded successfully!")
    print(f"Agent model: {agent.model}")

    env.close()


if __name__ == "__main__":
    print("CustomAgent Demo Scripts\n")
    print("Available examples:")
    print("1. Basic usage: Uncomment demo_basic_usage()")
    print("2. vs Random: Uncomment demo_vs_random()")
    print("3. Custom checkpoint: Uncomment demo_load_custom_checkpoint()\n")

    # Run basic demo
    try:
        demo_basic_usage()
    except Exception as e:
        print(f"Note: To run demos, you need to train the agent first.")
        print(f"Run: python train_custom_agent.py")
        print(f"\nError: {e}")

    # Uncomment to try other demos (requires trained checkpoint)
    # demo_vs_random()
    # demo_load_custom_checkpoint('./custom_agent/checkpoint.pth')
