"""
RNDAgent: DQN agent with Random Network Distillation for intrinsic exploration.
"""

from functools import partial
import jax
import jax.numpy as jnp
import flax
import optax
import time
import numpy as np
import os
from flax.training.train_state import TrainState
from utils.replay_buffer import ReplayBuffer
from exploration.rnd import RND
from networks.q_network import QNetwork


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class RNDAgent:
    """
    DQN + RND agent for intrinsic reward-based exploration.
    """

    def __init__(self, envs, config):
        """
        Initialize networks, optimizer, replay buffer and RND.

        Args:
            envs: Gymnax vectorized environment.
            config: Dictionary of hyperparameters.
        """
        self.envs = envs
        self.config = config
        self.tau = self.config.get("tau", 1.0)
        self.gamma = self.config["gamma"]

        self.q_network = QNetwork(action_dim=envs.single_action_space.n)
        self._q_apply_jit = jax.jit(lambda params, obs: self.q_network.apply({"params": params}, obs))

        obs_shape = envs.single_observation_space.shape
        dummy_obs = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
        self.rng = jax.random.PRNGKey(config["seed"])
        self.q_params = self.q_network.init(jax.random.PRNGKey(config["seed"] + 1), dummy_obs)

        self.optimizer = optax.adamw(config["learning_rate"])
        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.q_params["params"],
            tx=self.optimizer,
            target_params=self.q_params["params"],
        )
        self.opt_state = self.optimizer.init(self.q_state.params)

        self.rnd = RND(obs_shape, config)
        self.total_extrinsic_reward = 0.0
        self.total_timesteps = self.config["total_timesteps"]
        self.clip_extrinsic = config["clip_extrinsic_reward"]
        self.unique_state_ids = set()

        self.rb = ReplayBuffer(
            config["buffer_size"],
            envs.single_observation_space,
            envs.single_action_space,
        )
        self.log_info = {}

    @partial(jax.jit, static_argnums=0)
    def _select_action_jit(self, params, obs, key, temperature):
        """
        Boltzmann exploration action selection.

        Args:
            params: Q-network parameters.
            obs: Observations.
            key: JAX PRNG key.
            temperature (float): Temperature for softmax.

        Returns:
            np.ndarray: Selected actions.
        """
        q_values = self._q_apply_jit(params, obs)
        probs = jax.nn.softmax(q_values / temperature, axis=-1)
        actions = jax.random.categorical(key, jnp.log(probs), axis=-1)
        return actions

    def select_action(self, obs, global_step):
        """
        Select action with Boltzmann exploration.

        Args:
            obs: Observations.
            global_step (int): Current global step.

        Returns:
            np.ndarray: Selected actions.
        """
        self.rng, key = jax.random.split(self.rng)
        temperature = self.config.get("temperature", 1.0)
        actions = self._select_action_jit(self.q_state.params, obs, key, temperature)
        return np.atleast_1d(actions).astype(np.int32)

    def record_step(self, obs, next_obs, actions, rewards, dones, infos, global_step):
        """
        Store transition, compute rewards, track unique states.

        Args:
            obs: Observations.
            next_obs: Next observations.
            actions: Actions taken.
            rewards: Rewards received.
            dones: Done flags.
            infos: Env info dict.
            global_step (int): Current global step.
        """
        self.rnd.update_obs_stats(obs)
        self.total_extrinsic_reward += float(jnp.sum(rewards))

        intrinsic_reward = self.rnd.compute_intrinsic_reward(obs)
        extrinsic_coef = self.config.get("extrinsic_coef", 1.0)
        intrinsic_coef = self.config.get("intrinsic_coef", 1.0)
        total_reward = extrinsic_coef * rewards + intrinsic_coef * intrinsic_reward

        if self.clip_extrinsic:
            clipped_rewards = jnp.clip(rewards, -1.0, 1.0)
            total_reward = extrinsic_coef * clipped_rewards + intrinsic_coef * intrinsic_reward

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "state" in info:
                    self.unique_state_ids.add(info["state"])

        self.log_info.update({
            "rewards/extrinsic_mean": float(jnp.mean(rewards)),
            "rewards/intrinsic_mean": float(jnp.mean(intrinsic_reward)),
            "rewards/total_mean": float(jnp.mean(total_reward)),
            "rewards/max_extrinsic": float(jnp.max(rewards)),
            "rewards/max_intrinsic": float(jnp.max(intrinsic_reward)),
            "steps/global": global_step,
            "exploration/unique_states": len(self.unique_state_ids),
        })

        self.rb.add(obs, next_obs, actions, total_reward, dones, infos)

    def train_step(self, global_step):
        """
        Sample batch and update Q-network and RND.

        Args:
            global_step (int): Current global step.
        """
        if not self.rb.can_sample(self.config["batch_size"]):
            return

        batch = self.rb.sample(self.config["batch_size"])
        obs, next_obs, actions, rewards, dones = (
            batch.observations, batch.next_observations,
            batch.actions, batch.rewards, batch.dones
        )

        if global_step % self.config.get("predictor_update_freq", 1) == 0:
            rnd_loss = self.rnd.update(obs)
            self.log_info.update({"rnd/predictor_loss": float(rnd_loss)})

        loss, grad_mse, q_pred, td_target, new_params, new_opt_state = self.update(
            self.q_state.params, self.q_state.target_params, self.q_state.opt_state,
            obs, actions.squeeze(), next_obs, rewards.flatten(), dones.flatten()
        )
        self.q_state = self.q_state.replace(params=new_params, opt_state=new_opt_state)

        num_actions = self.envs.single_action_space.n
        one_hot_actions = jax.nn.one_hot(actions.squeeze(), num_actions)
        p_a_t = jnp.mean(one_hot_actions, axis=0)
        entropy = -jnp.sum(p_a_t * jnp.log(p_a_t + 1e-12))

        if global_step > 0:
            sample_efficiency = self.total_extrinsic_reward / global_step
            self.log_info.update({"charts/sample_efficiency": float(sample_efficiency)})

        self.log_info.update({
            "losses/td_loss": float(loss),
            "q/mean_q": float(jnp.mean(q_pred)),
            "q/mean_target": float(jnp.mean(td_target)),
            "q/grad_mse": float(grad_mse),
            "exploration/entropy": float(entropy),
        })

    def log_metrics(self, global_step, start_time):
        """
        Log metrics and steps-per-second.

        Args:
            global_step (int): Current global step.
            start_time (float): Start time in seconds.

        Returns:
            dict: Metrics to log.
        """
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def update_target_network(self):
        """
        Polyak averaging of target Q-network.
        """
        self.q_state = self.q_state.replace(
            target_params=optax.incremental_update(
                self.q_state.params, self.q_state.target_params, self.tau
            )
        )

    def save(self, path):
        """
        Save agent parameters to file.

        Args:
            path (str): Save file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "policy_params": self.q_state.params,
            "rnd_params": self.rnd.train_state.params,
        }
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(state))

    @partial(jax.jit, static_argnums=0)
    def update(self, params, target_params, opt_state, obs, actions, next_obs, rewards, dones):
        """
        Q-learning update with TD error.

        Args:
            params: Q-network parameters.
            target_params: Target Q-network parameters.
            opt_state: Optimizer state.
            obs: Observations.
            actions: Actions taken.
            next_obs: Next observations.
            rewards: Rewards received.
            dones: Done flags.

        Returns:
            tuple: Loss, grad MSE, Q-predictions, TD targets, new params, new opt_state.
        """
        def loss_fn(params):
            q_values = self._q_apply_jit(params, obs)
            q_pred = q_values[jnp.arange(q_values.shape[0]), actions]
            q_next = self.q_network.apply({"params": target_params}, next_obs)
            q_target = jnp.max(q_next, axis=-1)
            td_target = rewards + self.gamma * q_target * (1.0 - dones)
            loss = jnp.mean((q_pred - td_target) ** 2)
            return loss, (q_pred, td_target)

        (loss, (q_pred, td_target)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        squared_grad_norms = jax.tree_util.tree_map(lambda g: jnp.mean(g ** 2), grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(squared_grad_norms)
        grad_mse = jnp.mean(flat_grads)

        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return loss, grad_mse, q_pred, td_target, new_params, new_opt_state