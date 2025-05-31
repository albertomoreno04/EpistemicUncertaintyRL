import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import flax
import yaml
from environments.make_env import make_env, make_atari_env
from networks.q_network import QNetwork
from agents.rnd_agent import RNDAgent
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents import make_agent
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from tqdm.auto import tqdm
import wandb
import hashlib
import json



# ALGO LOGIC: initialize agent here:



class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def run_single_seed(config):
    config_for_group = {k: v for k, v in config.items() if k != "seed"}
    group_name = hashlib.md5(json.dumps(config_for_group, sort_keys=True).encode()).hexdigest()

    if config["track"]:
        run = wandb.init(config=config, group=group_name, reinit=True)
        config.update(dict(run.config))
        hidden_dim = config.get("hidden_dim")
        if hidden_dim is not None:
            config["predictor_hidden_dim"] = hidden_dim
            config["target_hidden_dim"] = hidden_dim

        # TRY NOT TO MODIFY: seeding
        random.seed(config["seed"])  # for python default library
        np.random.seed(config["seed"])  # for numpy
        key = jr.key(config["seed"])  # jax: sets up the key
        key, q_key = jr.split(key)  # splits the key in 2

        # env setup
        if "Montezuma" in config["env_id"] or "ALE/" in config["env_id"]:
            env_fns = [make_atari_env(config["env_id"], config["seed"], 0, config["capture_video"])]
        else:
            env_fns = [make_env(config["env_id"], config["seed"], 0, config["capture_video"])]

        envs = gym.vector.SyncVectorEnv(env_fns)
        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "Only discrete action space is supported"

        obs, _ = envs.reset(seed=config["seed"])

        agent = make_agent(config["agent_type"], envs, config)

        start_time = time.time()  # start timer for SPS (steps-per-second) computation
        last_mean_rs = 0  # the average reward, reporting purposes

        # TRY NOT TO MODIFY: start the game
        progress_bar = tqdm(range(config["total_timesteps"]))

        # For recording videos at 10% intervals
        video_interval = config["total_timesteps"] // 10
        video_checkpoints = set(
            i * video_interval for i in range(1, 11)
        )  # 10%, 20%, ..., 100%
        last_video_path = None

        for global_step in progress_bar:
            if config["track"]:
                wandb.log({})  # commit to wandb only if tracking

            actions = agent.select_action(obs, global_step)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # record rewards for reporting purposes
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

                if config["track"]:  # post updates into wandb
                    wandb.log(
                        {
                            "charts/episodic_return": jnp.array(rs).mean(),
                            "charts/episodic_length": jnp.array(ls).mean(),
                        },
                        commit=False,
                    )

            if global_step % 200 == 0:  # print pretty updates into tty
                progress_bar.set_description_str(f"Reward: {float(last_mean_rs):.2f}")

            # save data to reply buffer; handle `final_observation`
            agent.record_step(obs, next_obs, actions, rewards, terminations, infos, global_step)
            agent.train_step(global_step)

            # CRUCIAL step easy to overlook, moving to the new observations
            obs = next_obs

            if global_step % 500 == 0 and config.get("track", False):
                wandb.log(agent.log_metrics(global_step, start_time), commit=True)

            if global_step % config["target_network_frequency"] == 0:
                agent.update_target_network()

        if config["save_model"]:  # even more reporting
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            model_path = f"runs/{run_name}/{config['exp_name']}.nanodqn_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            progress_bar.write(f"Model saved to {model_path}")

            from cleanrl_utils.evals.dqn_jax_eval import evaluate

            epsilon_eval = 0.0 if config["agent_type"] == "rnd" else 0.05
            episodic_returns = evaluate(
                model_path,
                env_factory,
                config["env_id"],
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                epsilon=epsilon_eval,
            )
            if config["track"]:
                wandb.log({"eval/episodic_return_mean": episodic_returns.mean()})

        envs.close()


if __name__ == "__main__":

    with open("configs/rnd.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    run_single_seed(base_config)