from .epsilon_greedy_agent import EpsilonGreedyAgent
from .rnd_agent import RNDAgent
from .rnd_agent_deepsea import RNDDeepSeaAgent
from .epsilon_greedy_agent_deepsea import EpsilonGreedyAgentDeepSea

def make_agent(agent_type: str, envs, config: dict, env_params=None):
    if agent_type == "epsilon_greedy" and config["env_id"] == "DeepSea-bsuite":
        return EpsilonGreedyAgentDeepSea(envs, config, env_params)
    elif agent_type == "epsilon_greedy":
        return EpsilonGreedyAgent(envs, config)
    elif agent_type == "rnd" and config["env_id"] == "DeepSea-bsuite":
        return RNDDeepSeaAgent(envs, config, env_params)
    elif agent_type == "rnd" and config["env_id"]:
        return RNDAgent(envs, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")