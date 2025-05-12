from functools import partial

import jax
import jax.numpy as jnp
import chex
import flax
import optax
import numpy as np
import time
import os
from flax.training.train_state import TrainState
from utils.replay_buffer import ReplayBuffer
from exploration.drnd import DRND
from networks.rnd_networks import PolicyModel

class DRNDAgent:
    def __init__(self, envs, config):
        self.envs = envs
        self.config = config
        obs_shape = envs.single_observation_space.shape
        action_dim = envs.single_action_space.n

        self.key = jax.random.PRNGKey(config["seed"])
        self.policy = PolicyModel(obs_shape[0], action_dim)
        dummy_obs = jnp.zeros((1, obs_shape[0]))
        self.policy_params = self.policy.init(jax.random.PRNGKey(config["seed"] + 1), dummy_obs)

        self.optimizer = optax.adam(config["learning_rate"])
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy_params["params"],
            tx=self.optimizer,
        )
        self.policy_opt_state = self.optimizer.init(self.policy_state.params)

        self.drnd = DRND(obs_shape, config)

        self.rb = ReplayBuffer(
            config["buffer_size"],
            envs.single_observation_space,
            envs.single_action_space,
        )

        self.log_info = {}

    @partial(jax.jit, static_argnums=0)
    def _select_action_jit(self, params, obs, key):
        chex.assert_max_traces(1)
        logits = self.policy.apply({"params": params}, obs)
        action = jax.random.categorical(key, logits)
        return action

    def select_action(self, obs, global_step):
        self.key, subkey = jax.random.split(self.key)
        action = self._select_action_jit(self.policy_state.params, obs, subkey)
        action = jax.device_get(action)

        if isinstance(action, np.ndarray):
            action = action.item()

        return np.array([action])

    def record_step(self, obs, next_obs, actions, rewards, dones, infos, global_step, max_steps):
        intrinsic_reward = self.drnd.compute_intrinsic_reward(obs, global_step, max_steps)
        total_reward = rewards + self.config["intrinsic_reward_scale"] * intrinsic_reward
        self.rb.add(obs, next_obs, actions, total_reward, dones, infos)

    def train_step(self, global_step):
        if not self.rb.can_sample(self.config["batch_size"]):
            return
        if global_step % self.config["train_frequency"] != 0:
            return


        batch = self.rb.sample(self.config["batch_size"])
        obs, next_obs, actions, rewards, dones = (
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards,
            batch.dones,
        )

        self.policy_state, self.policy_opt_state, loss = self._policy_update_step(
            self.policy_state, self.policy_opt_state, obs, actions, rewards
        )

        # Also update the DRND predictor
        if global_step % self.config["predictor_update_freq"] == 0:
            self.drnd.train_state = self.drnd.update_step(self.drnd.train_state, obs)

        self.log_info = {
            "losses/policy_loss": float(loss),
            "drnd/mean_reward": float(jnp.mean(rewards)),
        }

    def log_metrics(self, global_step, start_time):
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def update_target_network(self):
        # DRND doesn't use a DQN target network.
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "policy_params": self.policy_state.params,
            "drnd_params": self.drnd.train_state.params,
        }
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(state))

    @partial(jax.jit, static_argnums=0)
    def _policy_update_step(self, policy_state, policy_opt_state, obs, actions, rewards):
        def loss_fn(params):
            logits = self.policy.apply({"params": params}, obs)
            log_probs = jax.nn.log_softmax(logits)
            selected_log_probs = jnp.take_along_axis(log_probs, actions, axis=1).squeeze()
            loss = -jnp.mean(selected_log_probs * rewards)
            return loss

        chex.assert_max_traces(1)
        loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
        updates, policy_opt_state = self.optimizer.update(grads, policy_opt_state)
        policy_state = policy_state.replace(params=optax.apply_updates(policy_state.params, updates))

        return policy_state, policy_opt_state, loss
