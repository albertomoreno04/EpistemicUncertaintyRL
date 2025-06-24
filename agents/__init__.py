"""
Factory for creating DQN or RND agents across different environments.
"""

from .epsilon_greedy_agent import EpsilonGreedyAgent
from .rnd_agent import RNDAgent
from .rnd_gymnax_agent import RNDGymnaxAgent
from .epsilon_greedy_gymnax_agent import EpsilonGreedyGymnaxAgent


def make_agent(agent_type: str, envs, config: dict, env_params=None):
    """
    Instantiate an agent based on type and environment.

    Args:
        agent_type (str): One of ["epsilon_greedy", "rnd"].
        envs: Environment object.
        config (dict): Experiment configuration.
        env_params: Optional environment parameters (Gymnax specific).

    Returns:
        An agent instance of the appropriate class.

    Raises:
        ValueError: If an unknown agent type is provided.
    """
    if agent_type == "epsilon_greedy" and config["env_id"] in ["DeepSea-bsuite", "MNISTBandit-bsuite"]:
        return EpsilonGreedyGymnaxAgent(envs, config, env_params)
    elif agent_type == "epsilon_greedy":
        return EpsilonGreedyAgent(envs, config)
    elif agent_type == "rnd" and config["env_id"] in ["DeepSea-bsuite", "MNISTBandit-bsuite"]:
        return RNDGymnaxAgent(envs, config, env_params)
    elif agent_type == "rnd":
        return RNDAgent(envs, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
