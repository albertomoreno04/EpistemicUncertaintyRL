from .epsilon_greedy_agent import EpsilonGreedyAgent
from .rnd_agent import RNDAgent

def make_agent(agent_type: str, envs, config: dict):
    if agent_type == "epsilon_greedy":
        return EpsilonGreedyAgent(envs, config)
    elif agent_type == "rnd":
        return RNDAgent(envs, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")