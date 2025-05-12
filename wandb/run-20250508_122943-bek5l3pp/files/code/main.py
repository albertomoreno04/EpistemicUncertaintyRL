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

with open("configs/drnd.yaml", "r") as f:
    config = yaml.safe_load(f)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = config["seed"]
    """seed of the experiment"""
    track: bool = config["track"]
    """if toggled, this experiment will be tracked with Weights and Biases; on by default"""
    wandb_project_name: str = config["wandb_project_name"]
    """the wandb's project name"""
    capture_video: bool = config["capture_video"]
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = config["save_model"]
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    agent_type: str = config["agent_type"]
    """type of agent used (e.g., epsilon_greedy, rnd)"""
    env_id: str = config["env_name"]
    """the id of the environment"""
    total_timesteps: int = config["total_timesteps"]
    """total timesteps of the experiments"""
    num_envs: int = config["num_envs"]
    """the number of parallel game environments"""
    buffer_size: int = config["buffer_size"]
    """the replay memory buffer size"""
    batch_size: int = config["batch_size"]
    """the batch size of sample from the reply memory"""


# ALGO LOGIC: initialize agent here:



class TrainState(TrainState):
    target_params: flax.core.FrozenDict



if __name__ == "__main__":

    args = tyro.cli(Args)
    assert args.num_envs == 1, "Vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}_{hex(int(time.time()) % 65536)}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)  # for python default library
    np.random.seed(args.seed)  # for numpy
    key = jr.key(args.seed)  # jax: sets up the key
    key, q_key = jr.split(key)  # splits the key in 2

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action space is supported"

    agent = make_agent(config["agent_type"], envs, config)

    start_time = time.time()  # start timer for SPS (steps-per-second) computation
    last_mean_rs = 0  # the average reward, reporting purposes

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    progress_bar = tqdm(range(args.total_timesteps))

    # For recording videos at 10% intervals
    video_interval = args.total_timesteps // 10
    video_checkpoints = set(
        i * video_interval for i in range(1, 11)
    )  # 10%, 20%, ..., 100%
    last_video_path = None

    for global_step in progress_bar:
        if args.track:
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

            if args.track:  # post updates into wandb
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
        agent.record_step(obs, next_obs, actions, rewards, terminations, infos, global_step, args.total_timesteps)
        agent.train_step(global_step)

        # CRUCIAL step easy to overlook, moving to the new observations
        obs = next_obs

        if global_step % 1000 == 0 and config.get("track", False):
            wandb.log(agent.log_metrics(global_step, start_time), commit=False)

        if hasattr(agent, "update_target_network"):
            if "target_network_frequency" in config:
                if global_step % config["target_network_frequency"] == 0:
                    agent.update_target_network()

    if args.save_model:  # even more reporting
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        model_path = f"runs/{run_name}/{args.exp_name}.nanodqn_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        progress_bar.write(f"Model saved to {model_path}")

        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        if args.track:
            wandb.log({"eval/episodic_return_mean": episodic_returns.mean()})

    envs.close()
