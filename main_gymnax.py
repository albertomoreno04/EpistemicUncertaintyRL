import random
import time
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm.auto import tqdm
import wandb
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
from agents import make_agent


def run_single_seed(config):
    """
    Runs one training loop with a single random seed.

    Args:
        config (dict): Experiment configuration dictionary containing:
            - env_id (str): Gymnax environment ID.
            - agent_type (str): Type of agent to use.
            - num_envs (int): Number of parallel environments.
            - total_timesteps (int): Total training timesteps.
            - learning_rate (float): Learning rate for optimizer.
            - buffer_size (int): Replay buffer size.
            - batch_size (int): Batch size for updates.
            - gamma (float): Discount factor.
            - tau (float): Target network smoothing factor.
            - target_network_frequency (int): Frequency to update target network.
            - track (bool): Whether to log to WandB.
            - seed (int): Random seed for reproducibility.
            - Other agent-specific parameters.
    """

    if config["track"]:
        run = wandb.init(config=config, reinit=True)
        config.update(dict(run.config))

    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    key = jr.PRNGKey(config["seed"])
    key, key_reset = jr.split(key, 2)

    # Vectorized environment functions
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    # Initialize Gymnax environment
    if config["env_id"] == "MNISTBandit-bsuite":
        basic_env, env_params = gymnax.make(config["env_id"])
    else:
        basic_env, env_params = gymnax.make(config["env_id"], size=config["env_size"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    obs, state = vmap_reset(config["num_envs"])(key_reset)

    # Initialize agent
    agent = make_agent(config["agent_type"], env, config, env_params)

    start_time = time.time()
    last_mean_rs = 0
    progress_bar = tqdm(range(config["total_timesteps"]))

    for global_step in progress_bar:
        if config["track"]:
            wandb.log({}, step=global_step)

        key, key_action, key_step = jr.split(key, 3)

        actions = agent.select_action(obs, global_step)
        next_obs, state, reward, done, info = vmap_step(config["num_envs"])(key_step, state, actions)

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
            }, step=global_step, commit=False)

        if global_step % 200 == 0:
            progress_bar.set_description_str(f"Reward: {float(last_mean_rs):.2f}")

        agent.record_step(obs, next_obs, actions, reward, done, info, global_step)
        agent.train_step(global_step)

        obs = next_obs

        if global_step % 500 == 0 and config.get("track", False):
            wandb.log(agent.log_metrics(global_step, start_time), step=global_step, commit=True)

        if global_step % config["target_network_frequency"] == 0:
            agent.update_target_network()


if __name__ == "__main__":
    """
    Loads config file and launches training loop.
    """
    with open("configs/DeepSea/rnd.yaml", "r") as f:  # Enter .yaml name
        base_config = yaml.safe_load(f)

    run_single_seed(base_config)
