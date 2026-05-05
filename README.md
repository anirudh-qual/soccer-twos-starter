# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.

## DQN Agent with Custom Observation Wrapper

This branch (`dqn_train`) contains a **Deep Q-Network (DQN) agent** implementation for Soccer-Twos, featuring a custom observation transformation wrapper for improved learning efficiency.

### Overview

The DQN agent (`custom_agent/`) is trained using Ray RLlib with the following key components:

- **Algorithm**: Deep Q-Network (DQN) with advanced techniques (Double DQN, 3-step returns, dueling networks, Prioritized Experience Replay, C51 distributional RL, Noisy Networks)
- **Observation Processing**: Custom `ObservationTransformWrapper` that converts high-dimensional Soccer Twos observations (200+ dims) into normalized, task-relevant features (40-80 dims)
- **Architecture**: Two-layer neural network with ReLU activation (256 units per layer)
- **Training**: Ray Tune with 100 iterations (100K environment timesteps)

### Training the DQN Agent

To train the DQN agent from scratch:

```bash
# Activate environment
conda activate soccertwos

# Run training script
python train_custom_agent.py
```

Default configuration:
- Learning rate: 1e-4
- Replay buffer size: 500K (with Prioritized Experience Replay)
- Batch size: 4096
- Target network update: 4000 steps
- Epsilon decay: 1.0 → 0.02 over 1M timesteps

### Evaluating the DQN Agent

**Test against random opponents:**
```bash
python evaluate_custom_agent.py
```

**Watch the trained agent play:**
```bash
python -m soccer_twos.watch -m custom_agent
```

**Demo with specific checkpoint:**
```bash
python custom_agent_demo.py
```

### Observation Wrapper Details

The `ObservationTransformWrapper` extracts task-relevant features:

- **Relative ball position & velocity** (normalized by field dimensions and max velocity)
- **Agent velocity & rotation** (normalized to [-1, 1])
- **Other players' relative positions & velocities**
- **Distance to ball** (Euclidean metric, normalized)
- **Angle to goal** (bearing angle, normalized to [-1, 1])

All features are clipped to [-1, 1] for stable learning. This normalized, egocentric representation enables better gradient flow and generalization across varying opponent positions.

### Pre-trained Checkpoint

A trained checkpoint is included at `custom_agent/checkpoint.pth` (iteration 600):
- Mean reward: ~0.05 to 0.15
- Training time: ~12 hours on 16 CPU cores
- Converged learning curve visible in `mean_reward_curve.png`

### Files & Structure

```
custom_agent/
├── agent.py                  # DQN agent class implementing AgentInterface
├── observation_wrapper.py    # Custom observation transformation logic
├── reward_shaping.py         # Reward modification utilities
├── checkpoint.pth            # Pre-trained model weights (iteration 600)
├── requirements.txt          # Agent-specific dependencies
└── README.md                 # Detailed agent documentation

train_custom_agent.py         # Training script with Ray RLlib configuration
evaluate_custom_agent.py      # Evaluation script (win rate vs policy)
custom_agent_demo.py          # Demo script for inference
training_curves_visualization.ipynb  # Jupyter notebook for metric visualization
report.tex                    # CORL 2026 paper on DQN + observation wrapper
```

### Research Highlights

This implementation was developed as part of a research comparison of two RL agents for Soccer Twos:
- **Agent 1 (Baseline)**: PPO with curriculum learning and reward shaping
- **Agent 2 (This branch)**: DQN with custom observation wrapper for feature engineering

The DQN approach leverages off-policy learning stability combined with engineered egocentric features to achieve competitive performance without dense reward shaping.

For detailed implementation notes and experimental results, see `report.tex`.


