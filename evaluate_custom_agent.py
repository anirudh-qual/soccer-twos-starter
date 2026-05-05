"""Evaluate CustomAgent against baseline agent modules over many matches."""

import argparse
import importlib
import os
from typing import Dict, List, Tuple, Type

import numpy as np
import soccer_twos
from soccer_twos import AgentInterface, EnvType

from custom_agent import CustomAgent


AGENT_CANDIDATE_CLASSNAMES = [
    "CustomAgent",
    "TeamAgent",
    "RandomAgent",
    "BaselineAgent",
    "Agent",
]


def _load_agent_class(module_name: str) -> Type[AgentInterface]:
    """Load an AgentInterface subclass from a module by common class names."""
    tried_modules = []

    # Try package import first, then fallback to `<package>.agent`.
    module_candidates = [module_name]
    if not module_name.endswith(".agent"):
        module_candidates.append(f"{module_name}.agent")

    module = None
    for candidate in module_candidates:
        tried_modules.append(candidate)
        try:
            module = importlib.import_module(candidate)
            break
        except ModuleNotFoundError:
            continue

    if module is None:
        raise ModuleNotFoundError(
            f"Could not import opponent module '{module_name}'. "
            f"Tried: {tried_modules}"
        )

    for class_name in AGENT_CANDIDATE_CLASSNAMES:
        if hasattr(module, class_name):
            candidate = getattr(module, class_name)
            if isinstance(candidate, type) and issubclass(candidate, AgentInterface):
                return candidate

    # Fallback: pick the first AgentInterface subclass found in module attributes.
    for _, candidate in vars(module).items():
        if (
            isinstance(candidate, type)
            and issubclass(candidate, AgentInterface)
            and candidate is not AgentInterface
        ):
            return candidate

    raise ValueError(
        f"Could not find an AgentInterface subclass in module '{module_name}'. "
        f"Tried imports: {tried_modules} | class names: {AGENT_CANDIDATE_CLASSNAMES}"
    )


def _split_teams(obs: Dict[int, np.ndarray]) -> Tuple[List[int], List[int]]:
    """Split player IDs into two teams (first half vs second half by sorted IDs)."""
    agent_ids = sorted(obs.keys())
    half = len(agent_ids) // 2
    return agent_ids[:half], agent_ids[half:]


def _select_actions(obs: Dict[int, np.ndarray], ids: List[int], agent: AgentInterface):
    """Request actions from an agent for a subset of players."""
    sub_obs = {agent_id: obs[agent_id] for agent_id in ids}
    return agent.act(sub_obs)


def evaluate_matchup(opponent_module: str, num_matches: int, max_steps: int, verbose: bool):
    """Evaluate CustomAgent against one opponent module."""
    print("\n" + "=" * 70)
    print(f"Evaluating custom_agent (Team 1) vs {opponent_module} (Team 2)")
    print(f"Matches: {num_matches} | Max steps/match: {max_steps}")
    print("=" * 70)

    unique_worker = (os.getpid() % 10000) + 1
    env = soccer_twos.make(
        variation=EnvType.multiagent_team,
        worker_id=unique_worker,
        base_port=10000 + unique_worker * 2,
    )
    # Warm-start environment process before initializing Ray-based opponent agents.
    # This avoids gRPC/fork crashes seen when env process creation happens after ray.init().
    _ = env.reset()

    custom_cls = CustomAgent
    opponent_cls = _load_agent_class(opponent_module)

    # Pass checkpoint_path to CustomAgent if available
    team1_agent = custom_cls(env, checkpoint_path=checkpoint_path)
    team2_agent = opponent_cls(env)

    stats = {
        "team1_wins": 0,
        "team2_wins": 0,
        "draws": 0,
        "team1_goals": [],
        "team2_goals": [],
    }

    for match_idx in range(num_matches):
        obs = env.reset()
        team1_ids, team2_ids = _split_teams(obs)
        team1_goals = 0
        team2_goals = 0

        done = False
        step = 0
        while not done and step < max_steps:
            actions_1 = _select_actions(obs, team1_ids, team1_agent)
            actions_2 = _select_actions(obs, team2_ids, team2_agent)

            all_actions = {**actions_1, **actions_2}
            obs, rewards, done, info = env.step(all_actions)

            for agent_id, reward in rewards.items():
                if reward >= 1.0:
                    if agent_id in team1_ids:
                        team1_goals += 1
                    elif agent_id in team2_ids:
                        team2_goals += 1

            step += 1

        if team1_goals > team2_goals:
            stats["team1_wins"] += 1
            result = "TEAM1_WIN"
        elif team2_goals > team1_goals:
            stats["team2_wins"] += 1
            result = "TEAM2_WIN"
        else:
            stats["draws"] += 1
            result = "DRAW"

        stats["team1_goals"].append(team1_goals)
        stats["team2_goals"].append(team2_goals)

        if verbose:
            print(
                f"Match {match_idx + 1:03d}: custom_agent {team1_goals} - "
                f"{team2_goals} {opponent_module} ({result})"
            )

    env.close()

    print("\nSummary")
    print("-" * 70)
    print(f"Team 1 wins (custom_agent): {stats['team1_wins']} ({stats['team1_wins']/num_matches*100:.1f}%)")
    print(f"Team 2 wins ({opponent_module}): {stats['team2_wins']} ({stats['team2_wins']/num_matches*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_matches*100:.1f}%)")
    print(f"Avg goals custom_agent: {np.mean(stats['team1_goals']):.2f}")
    print(f"Avg goals {opponent_module}: {np.mean(stats['team2_goals']):.2f}")
    print(
        "Goal difference (custom_agent - opponent): "
        f"{np.mean(np.array(stats['team1_goals']) - np.array(stats['team2_goals'])):.2f}"
    )
    print("-" * 70)

    return stats


def main(args):
    global checkpoint_path
    checkpoint_path = args.checkpoint
    for module_name in args.opponent_modules:
        try:
            evaluate_matchup(
                opponent_module=module_name,
                num_matches=args.num_matches,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
        except ModuleNotFoundError:
            print(f"\nSkipping '{module_name}' because the module was not found.")
        except Exception as exc:
            print(f"\nFailed evaluating vs '{module_name}': {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate custom_agent against one or more baseline modules"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file or directory for CustomAgent"
    )
    parser.add_argument(
        "--opponent-modules",
        nargs="+",
        default=["example_team_agent", "ceia_baseline_agent"],
        help=(
            "Python module names for opponent agents. "
            "Defaults to example_team_agent and ceia_baseline_agent."
        ),
    )
    parser.add_argument(
        "--num_matches",
        type=int,
        default=50,
        help="Number of matches per opponent (default: 50)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per match (default: 500)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-match score lines",
    )

    parsed_args = parser.parse_args()

    try:
        main(parsed_args)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
