## Self-Play Agent Training

Self-play training enables agents to improve by competing against themselves, allowing for continuous skill progression and emergent behaviors.

### Training Self-Play Agent

```bash
# Activate environment
conda activate soccertwos

# Run self-play training script
python train_ray_selfplay.py
```

**Key features:**
- Agents compete against themselves for continuous improvement
- Shared weight updates across both team members
- No external opponents required — purely self-competitive learning
- Natural curriculum-like difficulty progression as both agents improve
- Emerges diverse strategies from repeated interactions

### Running Self-Play Agent

**Watch self-play agent compete:**
```bash
python -m soccer_twos.watch -m selfplay_agent
```

**Compare self-play agent vs random opponent:**
```bash
python -m soccer_twos.watch -m1 selfplay_agent -m2 example_player_agent
```

**Evaluate self-play performance:**
```bash
python evaluate_custom_agent.py --agent selfplay_agent
```

### Self-Play Results

Self-play training produces agents with:
- **Better generalization** — improved performance against unseen opponents
- **Emergent strategies** — complex behaviors from self-interaction dynamics
- **Robust policies** — agents adapt to various playstyles
- **Scalability** — continuous improvement without external data requirements

Self-play is particularly effective for multi-agent environments like Soccer Twos where agent behavior continuously adapts and coevolves.


