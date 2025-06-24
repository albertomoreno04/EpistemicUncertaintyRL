import hashlib
from functools import partial
import jax
import jax.numpy as jnp
import flax
import optax
import time
import numpy as np
import os
from flax.training.train_state import TrainState
import wandb
from utils.replay_buffer import ReplayBuffer
from exploration.rnd_gymnax import RND
from networks.q_network import QNetwork
from gymnasium.spaces import Box, Discrete
from gymnax.environments.spaces import Box as GxBox, Discrete as GxDiscrete


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class RNDGymnaxAgent:
    """Agent with DQN + RND for DeepSea and MNISTBandit environments."""

    def convert_space(self, space):
        """Converts Gymnax space to Gymnasium space."""
        if isinstance(space, GxBox):
            return Box(low=float(space.low), high=float(space.high), shape=space.shape, dtype=np.float32)
        elif isinstance(space, GxDiscrete):
            return Discrete(space.n)
        else:
            raise NotImplementedError(f"Unsupported space: {type(space)}")

    def __init__(self, envs, config, env_params=None):
        """Initializes agent, networks, RND module, and replay buffer."""
        self.envs = envs
        self.config = config

        act_space = envs.action_space(env_params)
        self.single_action_space = self.convert_space(act_space)
        obs_space = envs.observation_space(env_params)
        self.single_observation_space = self.convert_space(obs_space)

        self.q_network = QNetwork(action_dim=self.single_action_space.n)

        if config["env_id"] == "MNISTBandit-bsuite":
            encoded_obs_dim = 784
        else:
            encoded_obs_dim = 2 * config["env_size"]

        self.rnd = RND((encoded_obs_dim,), config)
        self.dummy_obs = jnp.zeros((1, encoded_obs_dim))
        self.rb = ReplayBuffer(config["buffer_size"], self.single_observation_space, self.single_action_space)

        self._q_apply_jit = jax.jit(lambda params, obs: self.q_network.apply(
            {"params": params}, obs.reshape(obs.shape[0], -1))
        )

        self.rng = jax.random.PRNGKey(config["seed"])
        self.q_params = self.q_network.init(jax.random.PRNGKey(config["seed"] + 1), self.dummy_obs)

        self.optimizer = optax.adamw(config["learning_rate"])
        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.q_params["params"],
            tx=self.optimizer,
            target_params=self.q_params["params"],
        )
        self.opt_state = self.optimizer.init(self.q_state.params)

        self.tau = config.get("tau", 1.0)
        self.gamma = config["gamma"]

        self.total_extrinsic_reward = 0.0
        self.total_timesteps = config["total_timesteps"]
        self.clip_extrinsic = config["clip_extrinsic_reward"]

        self.unique_state_ids = set()
        self.log_info = {}

    def encode_obs(self, obs):
        """Encodes DeepSea grid observation to one-hot position."""
        n = self.config["env_size"]
        batch_size = obs.shape[0]
        obs_reshaped = obs.reshape((batch_size, n, n))

        row_idx = jnp.argmax(jnp.any(obs_reshaped == 1, axis=2), axis=1)
        col_idx = jnp.argmax(jnp.any(obs_reshaped == 1, axis=1), axis=1)

        row_onehot = jax.nn.one_hot(row_idx, n)
        col_onehot = jax.nn.one_hot(col_idx, n)

        return jnp.concatenate([row_onehot, col_onehot], axis=-1)

    @partial(jax.jit, static_argnums=0)
    def _select_action_jit(self, params, obs, key, temperature):
        """Selects action with Boltzmann exploration."""
        if self.config["env_id"] == "DeepSea-bsuite":
            obs = self.encode_obs(obs)
        q_values = self._q_apply_jit(params, obs)
        probs = jax.nn.softmax(q_values / temperature, axis=-1)
        return jax.random.categorical(key, jnp.log(probs), axis=-1)

    def select_action(self, obs, global_step):
        """Samples action for current observation."""
        self.rng, key = jax.random.split(self.rng)
        return np.atleast_1d(self._select_action_jit(
            self.q_state.params, obs, key, self.config.get("temperature", 1.0))
        ).astype(np.int32)

    def record_step(self, obs, next_obs, actions, rewards, dones, infos, global_step):
        """Logs transition, computes intrinsic reward, adds to buffer."""
        if self.config["env_id"] == "DeepSea-bsuite":
            encoded_obs = self.encode_obs(obs)
        else:
            encoded_obs = obs

        self.rnd.update_obs_stats(encoded_obs)
        self.total_extrinsic_reward += float(jnp.sum(rewards))

        intrinsic_reward = self.rnd.compute_intrinsic_reward(encoded_obs)
        total_reward = (self.config.get("extrinsic_coef", 1.0) * rewards
                        + self.config.get("intrinsic_coef", 1.0) * intrinsic_reward)

        if self.clip_extrinsic:
            clipped_rewards = jnp.clip(rewards, -1.0, 1.0)
            total_reward = (self.config.get("extrinsic_coef", 1.0) * clipped_rewards +
                            self.config.get("intrinsic_coef", 1.0) * intrinsic_reward)

        try:
            obs_np = np.asarray(obs)
            for ob in obs_np:
                self.unique_state_ids.add(hashlib.sha1(ob.tobytes()).hexdigest())
        except Exception as e:
            print(f"Warning hashing obs: {e}")

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
        wandb.log(self.log_info, step=global_step)

    def train_step(self, global_step):
        """Performs Q-learning and RND predictor update."""
        if not self.rb.can_sample(self.config["batch_size"]):
            return

        batch = self.rb.sample(self.config["batch_size"])

        if global_step % self.config.get("predictor_update_freq", 1) == 0:
            if self.config["env_id"] == "DeepSea-bsuite":
                encoded_obs = self.encode_obs(batch.observations)
            else:
                encoded_obs = batch.observations
            new_train_state, rnd_loss = self.rnd.update(encoded_obs)
            self.rnd.train_state = new_train_state
            self.log_info.update({"rnd/predictor_loss": float(rnd_loss)})

        loss, grad_mse, q_pred, td_target, new_params, new_opt_state = self.update(
            self.q_state.params,
            self.q_state.target_params,
            self.q_state.opt_state,
            self.encode_obs(batch.observations) if self.config["env_id"] == "DeepSea-bsuite" else batch.observations,
            batch.actions.squeeze(),
            self.encode_obs(batch.next_observations) if self.config[
                                                            "env_id"] == "DeepSea-bsuite" else batch.next_observations,
            batch.rewards.flatten(),
            batch.dones.flatten()
        )
        self.q_state = self.q_state.replace(params=new_params, opt_state=new_opt_state)

        num_actions = self.single_action_space.n
        one_hot_actions = jax.nn.one_hot(batch.actions.squeeze(), num_actions)
        p_a_t = jnp.mean(one_hot_actions, axis=0)
        entropy = -jnp.sum(p_a_t * jnp.log(p_a_t + 1e-12))

        self.log_info.update({
            "losses/td_loss": float(loss),
            "q/mean_q": float(jnp.mean(q_pred)),
            "q/mean_target": float(jnp.mean(td_target)),
            "q/grad_mse": float(grad_mse),
            "exploration/entropy": float(entropy),
        })

    def log_metrics(self, global_step, start_time):
        """Returns dictionary of current tracked metrics."""
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        if global_step > 0:
            self.log_info["charts/sample_efficiency"] = self.total_extrinsic_reward / global_step
        return self.log_info

    def update_target_network(self):
        """Updates target network with smoothing."""
        self.q_state = self.q_state.replace(
            target_params=optax.incremental_update(
                self.q_state.params, self.q_state.target_params, self.tau
            )
        )

    def save(self, path):
        """Saves agent and RND parameters."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes({
                "policy_params": self.q_state.params,
                "rnd_params": self.rnd.train_state.params,
            }))

    @partial(jax.jit, static_argnums=0)
    def update(self, params, target_params, opt_state, obs, actions, next_obs, rewards, dones):
        """Performs a Q-learning update step."""
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
