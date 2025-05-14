import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import flax
import yaml
from environments.make_env import make_env
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



# ALGO LOGIC: initialize agent here:



class TrainState(TrainState):
    target_params: flax.core.FrozenDict



if __name__ == "__main__":

    with open("configs/rnd.yaml", "r") as f:
        config = yaml.safe_load(f)

    with wandb.init(config=config) as run:
        config.update(dict(run.config))
        config["predictor_hidden_dim"] = config["hidden_dim"]
        config["target_hidden_dim"] = config["hidden_dim"]

        run_name = f"{config['env_id']}_{hex(int(time.time()) % 65536)}"

        if config["track"]:
            wandb.run.name = run_name
            wandb.run.save()

        # TRY NOT TO MODIFY: seeding
        random.seed(config["seed"])  # for python default library
        np.random.seed(config["seed"])  # for numpy
        key = jr.key(config["seed"])  # jax: sets up the key
        key, q_key = jr.split(key)  # splits the key in 2

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(config["env_id"], config["seed"] + i, i, config["capture_video"], run_name)
                for i in range(config["num_envs"])
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "Only discrete action space is supported"

        agent = make_agent(config["agent_type"], envs, config)

        start_time = time.time()  # start timer for SPS (steps-per-second) computation
        last_mean_rs = 0  # the average reward, reporting purposes

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=config["seed"])
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
            agent.record_step(obs, next_obs, actions, rewards, terminations, infos, global_step, config["total_timesteps"])
            agent.train_step(global_step)

            # CRUCIAL step easy to overlook, moving to the new observations
            obs = next_obs

            if global_step % 1000 == 0 and config.get("track", False):
                wandb.log(agent.log_metrics(global_step, start_time), commit=False)

            if hasattr(agent, "update_target_network"):
                if "target_network_frequency" in config:
                    if global_step % config["target_network_frequency"] == 0:
                        agent.update_target_network()

        if config["save_model"]:  # even more reporting
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            model_path = f"runs/{run_name}/{config['exp_name']}.nanodqn_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            progress_bar.write(f"Model saved to {model_path}")

            from cleanrl_utils.evals.dqn_jax_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                config["env_id"],
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                epsilon=0.05,
            )
            if config["track"]:
                wandb.log({"eval/episodic_return_mean": episodic_returns.mean()})

        envs.close()
