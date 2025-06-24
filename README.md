# EpistemicUncertaintyRL

**EpistemicUncertaintyRL** is a JAX-based reinforcement learning framework designed to investigate exploration strategies under epistemic uncertainty. The framework implements both classic and intrinsic motivation methods, including Random Network Distillation (RND) and epsilon-greedy exploration, for Gymnasium and Gymnax environments.

The codebase emphasizes clean design, efficient JAX execution, and reproducibility, with integrated experiment tracking via Weights & Biases (W&B).

---

## Features

- DQN agent with optional Random Network Distillation (RND)  
- Epsilon-greedy baseline for exploration  
- Support for Gymnax environments like DeepSea and MNISTBandit  
- Classic Gymnasium environments such as Acrobot and CartPole  
- Vectorized environment execution (Gymnax & Gymnasium)  
- Seamless Weights & Biases logging  
- Reproducible, seed-controlled experiments  
- Clean, modular JAX + Flax implementation  

---

## Installation

1. Set up a virtual environment with [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate
```

2.	Install dependencies:

```bash
uv sync
```

3.	Authenticate with Weights & Biases for experiment tracking:

```bash
wandb login
```

## Project Structure
```bash
├── agents/                 # Agent implementations (RND, epsilon-greedy)
├── exploration/            # RND modules and exploration logic
├── environments/           # Gymnasium & Gymnax wrappers
├── networks/               # Q-network and RND network definitions
│   ├── q_network.py
│   └── rnd_networks.py
├── utils/                  # Replay buffer and utilities
│   └── replay_buffer.py
├── configs/                # Experiment configurations
│   ├── Acrobot/
│   │   ├── rnd.yaml
│   │   └── epsilon_greedy.yaml
│   ├── CartPole/
│   │   ├── rnd.yaml
│   │   └── epsilon_greedy.yaml
│   ├── DeepSea/
│   │   ├── rnd.yaml
│   │   └── epsilon_greedy.yaml
│   └── MNISTBandit/
│       ├── rnd.yaml
│       └── epsilon_greedy.yaml
├── main.py                 # Entry point for Gymnasium experiments
├── main_gymnax.py          # Entry point for Gymnax experiments
└── requirements.txt
```

## Running Experiments

### Gymnasium Environments (e.g., Acrobot, CartPole)
```
uv run main.py
```
**Reminder**: Make sure to modify main.py to load the correct YAML file from configs/Acrobot/ or configs/CartPole/ before running.

### Gymnax Environments (e.g., DeepSea, MNISTBandit)
```
uv run main_gymnax.py
```
**Reminder**: Update main_gymnax.py to load the correct YAML file from configs/DeepSea/ or configs/MNISTBandit/ before running.

## Example Configuration

Example RND configuration for DeepSea:
```
env_id: "DeepSea-bsuite"
agent_type: rnd
total_timesteps: 250000
learning_rate: 0.001
intrinsic_coef: 1.0
extrinsic_coef: 2.0
buffer_size: 50000
batch_size: 16
predictor_hidden_dim: 256
target_hidden_dim: 256
```
See the configs/ directory for fully tunable experiment setups for each environment.

## Acknowledgements

This project builds upon:

- [CleanRL](https://docs.cleanrl.dev/rl-algorithms/dqn/) — High-quality, reproducible RL baselines.
- [Gymnax](https://github.com/RobertTLange/gymnax) — JAX-native environments for RL research.
- [Flax](https://github.com/google/flax) — Neural network library for JAX.
- [Optax](https://github.com/deepmind/optax) — Gradient processing and optimization library for JAX.
- [Weights & Biases (W&B)](https://wandb.ai/site) — Experiment tracking and visualization.

This code was developed as part of a Bachelor Thesis / Research Project (2025) for the [TU Delft CSE Research Project](https://github.com/TU-Delft-CSE/Research-Project).