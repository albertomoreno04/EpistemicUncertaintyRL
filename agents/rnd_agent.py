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
from networks.rnd_networks import PolicyModel


class RNDAgent:
    def __init__(self, envs, config):
        self.envs = envs
        self.config = config
        obs_shape = envs.single_observation_space.shape
        action_dim = envs.single_action_space.n


        self.policy = PolicyModel(obs_shape[0], action_dim)
        dummy_obs = jnp.zeros((1, obs_shape[0]))
        self.rng = jax.random.PRNGKey(config["seed"])
        self.policy_params = self.policy.init(jax.random.PRNGKey(config["seed"] + 1), dummy_obs)

        self.optimizer = optax.adam(config["learning_rate"])
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy_params["params"],
            tx=self.optimizer,
        )
        self.policy_opt_state = self.optimizer.init(self.policy_state.params)

        self.rnd = RND(obs_shape, config)

        self.rb = ReplayBuffer(
            config["buffer_size"],
            envs.single_observation_space,
            envs.single_action_space,
        )

        self.log_info = {}

    @partial(jax.jit, static_argnums=0)
    def _select_action_jit(self, params, obs, key):
        logits = self.policy.apply({"params": params}, obs)
        action = jax.random.categorical(key, logits)
        return action

    def select_action(self, obs, global_step):
        self.rng, key = jax.random.split(self.rng)
        action = self._select_action_jit(self.policy_state.params, obs, key)
        action = jax.device_get(action)

        if isinstance(action, np.ndarray):
            action = action.item()

        return np.array([action])

    def record_step(self, obs, next_obs, actions, rewards, dones, infos, global_step, max_steps):
        # Compute intrinsic reward
        intrinsic_reward = self.rnd.compute_intrinsic_reward(obs)
        total_reward = rewards + self.config["intrinsic_coef"] * intrinsic_reward

        self.rb.add(obs, next_obs, actions, total_reward, dones, infos)

    def train_step(self, global_step):
        if not self.rb.can_sample(self.config["batch_size"]):
            return

        batch = self.rb.sample(self.config["batch_size"])
        obs, next_obs, actions, rewards, dones = (
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards,
            batch.dones,
        )

        # Policy gradient update (placeholder, customize for PPO, A2C, etc.)
        loss, new_params, new_opt_state = self._policy_update(
            self.policy_state.params, self.policy_opt_state, obs, actions, rewards
        )
        self.policy_state = self.policy_state.replace(params=new_params)
        self.policy_opt_state = new_opt_state

        self.log_info = {
            "losses/policy_loss": float(loss),
            "rnd/mean_reward": float(jnp.mean(rewards)),
        }

    def log_metrics(self, global_step, start_time):
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def update_target_network(self):
        # RND does not need a target network update here
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "policy_params": self.policy_state.params,
            "rnd_params": self.rnd.train_state.params,
        }
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(state))

    @partial(jax.jit, static_argnums=0)
    def _policy_update(self, params, policy_opt_state, obs, actions, rewards):
        def loss_fn(params):
            logits = self.policy.apply({"params": params}, obs)
            log_probs = jax.nn.log_softmax(logits)
            selected_log_probs = jnp.take_along_axis(log_probs, actions, axis=1).squeeze()
            return -jnp.mean(selected_log_probs * rewards)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, policy_opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state