import os
import random
import time
import hashlib
import json
from dataclasses import dataclass

import flax
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from tqdm.auto import tqdm
import wandb
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax

from networks.q_network import QNetwork
from agents import make_agent
from utils.gymnax_wrapper import GymnaxWrapper


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def run_single_seed(config):
    # Generate a unique group name for logging
    config_for_group = {k: v for k, v in config.items() if k != "seed"}
    group_name = hashlib.md5(json.dumps(config_for_group, sort_keys=True).encode()).hexdigest()

    if config["track"]:
        run = wandb.init(config=config, group=group_name, reinit=True)
        config.update(dict(run.config))

    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    key = jr.PRNGKey(config["seed"])
    key, key_reset = jr.split(key, 2)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    # Initialize Gymnax environment wrapper with vectorization
    basic_env, env_params = gymnax.make(config["env_id"], size=config["env_size"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    # Reset environments
    obs, state = vmap_reset(config["num_envs"])(key_reset)

    # Initialize agent
    agent = make_agent(config["agent_type"], env, config, env_params)

    start_time = time.time()
    last_mean_rs = 0
    progress_bar = tqdm(range(config["total_timesteps"]))

    for global_step in progress_bar:
        if config["track"]:
            wandb.log({})

        key, key_action, key_step = jr.split(key, 3)

        row_indices = state.env_state.row
        col_indices = state.env_state.column
        num_columns = config["env_size"]

        row_one_hot = jax.nn.one_hot(row_indices.astype(jnp.int32), num_columns)
        col_one_hot = jax.nn.one_hot(col_indices.astype(jnp.int32), num_columns)

        pos_encoding = jnp.concatenate([row_one_hot, col_one_hot], axis=-1)

        actions = agent.select_action(pos_encoding, global_step)
        next_obs, state, reward, done, info = vmap_step(config["num_envs"])(
            key_step, state, actions
        )

        row_indices_next = state.env_state.row
        col_indices_next = state.env_state.column

        row_one_hot_next = jax.nn.one_hot(row_indices_next.astype(jnp.int32), num_columns)
        col_one_hot_next = jax.nn.one_hot(col_indices_next.astype(jnp.int32), num_columns)

        pos_encoding_next = jnp.concatenate([row_one_hot_next, col_one_hot_next], axis=-1)

        # Logging episodic returns
        rs, ls = [], []
        rs.append(info["returned_episode_returns"].mean())
        ls.append(info["returned_episode_returns"].mean())

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

        agent.record_step(pos_encoding, pos_encoding_next, actions, reward, done, info, global_step)
        agent.train_step(global_step)

        obs = next_obs

        if global_step % 500 == 0 and config.get("track", False):
            wandb.log(agent.log_metrics(global_step, start_time), commit=True)

        if global_step % config["target_network_frequency"] == 0:
            agent.update_target_network()


if __name__ == "__main__":
    with open("configs/rnd.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    run_single_seed(base_config)
