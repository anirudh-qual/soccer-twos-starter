# CustomAgent - DQN with Observation Transform

## Agent Information

**Agent Name:** CustomAgent  
**Algorithm:** Deep Q-Network (DQN)  
**Authors:** [Your Name] <[your.email@gatech.edu](mailto:your.email@gatech.edu)>

## Overview

CustomAgent uses a DQN policy trained on single-player `team_vs_policy` rollouts and executed for both players on a team at inference time.

This version removes the PPO/self-play pipeline and replaces it with a simpler DQN setup.

## Observation Changes

Training and inference use the same transformed observation representation:

1. Keep every second feature from the original observation vector.
2. Append four global statistics: mean, std, max, min.

This produces a compact and stable observation vector while preserving coarse scene information.

## Files

- `agent.py`: DQN inference agent and Q-network definition.
- `observation_wrapper.py`: Observation transform logic and training-time env wrapper.
- `checkpoint.pth`: Trained model weights (you provide this after training/export).

## Training

Run:

```bash
python train_custom_agent.py
```

Resume:

```bash
python train_custom_agent.py --restore <checkpoint_path>
```

The script trains with Ray RLlib DQN and stores results in `ray_results/DQN_CustomAgent_ObsTransform`.

## Inference

`CustomAgent` expects `checkpoint.pth` in this folder by default.

The checkpoint can be either:

- Raw model state dict
- Dict with key `model_state_dict`

## Notes

- Action space is flattened for DQN argmax selection and mapped back to MultiDiscrete actions.
- Keep the observation transform identical between training and inference to avoid policy mismatch.
