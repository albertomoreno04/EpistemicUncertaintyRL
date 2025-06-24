import os
import random
import time
import gymnasium as gym
import flax
import yaml
from environments.make_env import make_env, make_atari_env
from networks.q_network import QNetwork
from agents import make_agent
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.training.train_state import TrainState
from tqdm.auto import tqdm
import wandb
import hashlib
import json


class TrainState(TrainState):
    """Extension of TrainState to store target network parameters."""
    target_params: flax.core.FrozenDict


def initialize_obs_stats(rnd_module, envs, num_steps=1000):
    """
    Initializes running statistics for RND by interacting with the environment.

    Args:
        rnd_module: RND module with update_obs_stats method.
        envs: Gym vectorized environment.
        num_steps (int): Number of steps to use for initialization.
    """
    obs, _ = envs.reset()
    for _ in range(num_steps):
        random_actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_obs, _, terminations, truncations, infos = envs.step(random_actions)
        rnd_module.update_obs_stats(obs)
        obs = next_obs


def run_single_seed(config):
    """
    Runs a full training loop for a given configuration and random seed.

    Args:
        config (dict): Dictionary containing experiment configuration.
    """
    config_for_group = {k: v for k, v in config.items() if k != "seed"}
    group_name = hashlib.md5(json.dumps(config_for_group, sort_keys=True).encode()).hexdigest()

    if config["track"]:
        run = wandb.init(config=config, group=group_name, reinit=True)
        config.update(dict(run.config))

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    key = jr.key(config["seed"])
    key, q_key = jr.split(key)

    env_fns = [make_env(config["env_id"], config["seed"], 0, config["capture_video"])]
    envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    obs, _ = envs.reset(seed=config["seed"])
    agent = make_agent(config["agent_type"], envs, config)

    start_time = time.time()
    last_mean_rs = 0

    progress_bar = tqdm(range(config["total_timesteps"]))

    video_interval = config["total_timesteps"] // 10
    video_checkpoints = set(i * video_interval for i in range(1, 11))
    last_video_path = None

    for global_step in progress_bar:
        if config["track"]:
            wandb.log({})

        actions = agent.select_action(obs, global_step)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        rs, ls = [], []
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    rs.append(info["episode"]["r"])
                    ls.append(info["episode"]["l"])

            current_mean = jnp.array(rs).mean()
            if last_mean_rs == 0:
                last_mean_rs = current_mean
            last_mean_rs = current_mean * 0.01 + 0.99 * last_mean_rs

            if config["track"]:
                wandb.log({
                    "charts/episodic_return": jnp.array(rs).mean(),
                    "charts/episodic_length": jnp.array(ls).mean(),
                }, commit=False)

        if global_step % 200 == 0:
            progress_bar.set_description_str(f"Reward: {float(last_mean_rs):.2f}")

        agent.record_step(obs, next_obs, actions, rewards, terminations, infos, global_step)
        agent.train_step(global_step)

        obs = next_obs

        if global_step % 500 == 0 and config.get("track", False):
            wandb.log(agent.log_metrics(global_step, start_time), commit=True)

        if global_step % config["target_network_frequency"] == 0:
            agent.update_target_network()

    envs.close()


if __name__ == "__main__":
    """
    Main entry point for running a training experiment.

    Loads configuration from YAML and starts training.
    """
    with open("configs/Acrobot/rnd.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    run_single_seed(base_config)