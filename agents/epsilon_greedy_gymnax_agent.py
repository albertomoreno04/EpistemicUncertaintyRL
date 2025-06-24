"""
EpsilonGreedyGymnaxAgent: DQN agent with epsilon-greedy exploration for Gymnax environments.
"""

import hashlib
import random
from functools import partial
import jax
from networks.q_network import QNetwork
import time
import optax
import flax
from flax.training.train_state import TrainState
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from utils.replay_buffer import ReplayBuffer
from gymnasium.spaces import Box, Discrete
from gymnax.environments.spaces import Box as GxBox, Discrete as GxDiscrete


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class EpsilonGreedyGymnaxAgent:
    """
    Deep Q-Network agent with epsilon-greedy exploration for Gymnax vectorized environments.
    """

    def convert_space(self, space):
        """
        Convert Gymnax spaces to Gymnasium spaces.

        Args:
            space: Gymnax Box or Discrete space.

        Returns:
            Gymnasium Box or Discrete space.
        """
        if isinstance(space, GxBox):
            return Box(low=float(space.low), high=float(space.high), shape=space.shape, dtype=np.float32)
        elif isinstance(space, GxDiscrete):
            return Discrete(space.n)
        else:
            raise NotImplementedError(f"Unsupported space: {type(space)}")

    def __init__(self, envs, config, env_params):
        """
        Initialize agent, Q-network, optimizer, and replay buffer.

        Args:
            envs: Gymnax vectorized environment.
            config: Dictionary of hyperparameters.
            env_params: Parameters specific to Gymnax environment.
        """
        self.envs = envs
        self.config = config
        self.global_step = 0

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        key = jr.key(config["seed"])
        key, q_key = jr.split(key)

        self.env_params = env_params

        act_space = envs.action_space(env_params)
        self.single_action_space = self.convert_space(act_space)
        obs_space = envs.observation_space(env_params)
        self.single_observation_space = self.convert_space(obs_space)

        self.q_network = QNetwork(action_dim=self.single_action_space.n)
        dummy_obs = jnp.zeros((1, np.prod(obs_space.shape)))
        self.q_params = self.q_network.init(jax.random.PRNGKey(config["seed"] + 1), dummy_obs)

        self.optimizer = optax.adamw(config["learning_rate"])
        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.q_params["params"],
            target_params=self.q_params["params"],
            tx=self.optimizer,
        )

        self._q_apply_jit = jax.jit(lambda params, obs: self.q_network.apply(
            {"params": params}, obs.reshape(obs.shape[0], -1))
        )

        self.opt_state = self.optimizer.init(self.q_state.params)

        self.rb = ReplayBuffer(
            config["buffer_size"],
            self.single_observation_space,
            self.single_action_space,
        )

        self.train_frequency = config["train_frequency"]
        self.epsilon = config["epsilon"]
        self.epsilon_final = config["epsilon_final"]
        self.exploration_fraction = config["exploration_fraction"]
        self.total_timesteps = config["total_timesteps"]
        self.tau = config["tau"]

        self.log_info = {}
        self.rewards = 0.0
        self.unique_state_ids = set()

    @staticmethod
    def linear_schedule(start_e, end_e, duration, t):
        """
        Linear schedule for decaying epsilon.

        Args:
            start_e (float): Initial epsilon.
            end_e (float): Final epsilon.
            duration (int): Number of steps to anneal over.
            t (int): Current global step.

        Returns:
            float: Epsilon at current step.
        """
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def train_step(self, global_step):
        """
        Sample from replay buffer and perform one Q-network update.

        Args:
            global_step (int): Current global step.
        """
        if not self.rb.can_sample(self.config["batch_size"]):
            return

        batch = self.rb.sample(self.config["batch_size"])

        loss, grad_mse, q_pred, td_target, self.q_state = self.update(
            self.q_state,
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.rewards.flatten(),
            batch.dones.flatten(),
        )

        num_actions = self.single_action_space.n
        one_hot_actions = jax.nn.one_hot(batch.actions.squeeze(), num_actions)
        p_a_t = jnp.mean(one_hot_actions, axis=0)
        entropy = -jnp.sum(p_a_t * jnp.log(p_a_t + 1e-12))

        self.log_info = {
            "losses/td_loss": jax.device_get(loss),
            "q/mean_q": float(jnp.mean(q_pred)),
            "q/mean_target": float(jnp.mean(td_target)),
            "q/grad_mse": float(grad_mse),
            "exploration/entropy": float(entropy),
            "exploration/extrinsic_mean": float(jnp.mean(batch.rewards)),
            "rewards/total_mean": float(jnp.mean(batch.rewards)),
            "charts/sample_efficiency": self.rewards / global_step,
            "rewards/max_extrinsic": float(jnp.max(batch.rewards)),
            "exploration/episode_count": float(jnp.sum(batch.dones)),
            "exploration/unique_states": len(self.unique_state_ids),
            "steps/global": global_step,
        }

    def update_target_network(self):
        """
        Polyak averaging update of the target network.
        """
        self.q_state = self.q_state.replace(
            target_params=optax.incremental_update(
                self.q_state.params, self.q_state.target_params, self.tau
            )
        )

    def record_step(self, obs, next_obs, actions, rewards, terminations, infos, global_step):
        """
        Store transition in replay buffer and track unique states.

        Args:
            obs: Current observations.
            next_obs: Next observations.
            actions: Actions taken.
            rewards: Rewards received.
            terminations: Done flags.
            infos: Environment info dict.
            global_step (int): Current global step.
        """
        try:
            obs_np = np.asarray(obs)
            for ob in obs_np:
                obs_hash = hashlib.sha1(ob.tobytes()).hexdigest()
                self.unique_state_ids.add(obs_hash)
        except Exception as e:
            print(f"Warning: Failed to hash obs for unique states: {e}")

        if global_step > 0:
            self.rewards += float(jnp.sum(rewards))

        self.rb.add(obs, next_obs, actions, rewards, terminations, infos)

    def log_metrics(self, global_step, start_time):
        """
        Return current metrics for logging.

        Args:
            global_step (int): Current global step.
            start_time (float): Start time in seconds.

        Returns:
            dict: Metrics to log.
        """
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def select_action(self, obs, global_step):
        """
        Select action with epsilon-greedy exploration.

        Args:
            obs: Current observations.
            global_step (int): Current global step.

        Returns:
            np.ndarray: Selected actions.
        """
        epsilon = self.linear_schedule(
            self.epsilon, self.epsilon_final,
            self.exploration_fraction * self.total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = np.array([self.single_action_space.sample() for _ in range(self.config["num_envs"])])
        else:
            q_values = self._q_apply_jit(self.q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        return actions

    @partial(jax.jit, static_argnums=0)
    def update(self, q_state, observations, actions, next_observations, rewards, dones):
        """
        Compute Q-learning loss and apply gradient update.

        Args:
            q_state (TrainState): Current Q-network state.
            observations: Current observations.
            actions: Actions taken.
            next_observations: Next observations.
            rewards: Rewards received.
            dones: Done flags.

        Returns:
            tuple: Loss, grad norm, Q-values, TD-targets, updated state.
        """
        def mse_loss(params):
            q_next_target = self.q_network.apply({"params": q_state.target_params}, next_observations)
            q_next_target = jnp.max(q_next_target, axis=-1)
            td_target = rewards + (1 - dones) * self.config["gamma"] * q_next_target

            q_pred = self.q_network.apply({"params": params}, observations)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            return jnp.mean((q_pred - td_target) ** 2), (q_pred, td_target)

        (loss, (q_pred, td_target)), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)

        squared_grad_norms = jax.tree_util.tree_map(lambda g: jnp.mean(g ** 2), grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(squared_grad_norms)
        grad_mse = jnp.mean(flat_grads)

        q_state = q_state.apply_gradients(grads=grads)
        return loss, grad_mse, q_pred, td_target, q_state