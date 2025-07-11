"""
Epsilon-greedy DQN agent for Gymnasium environments.
"""

import hashlib
import random
from functools import partial
import jax
import jax.numpy as jnp
from networks.q_network import QNetwork
import time
import optax
import flax
from flax.training.train_state import TrainState
import numpy as np
import jax.random as jr
from utils.replay_buffer import ReplayBuffer


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class EpsilonGreedyAgent:
    """
    DQN agent with epsilon-greedy exploration.
    """

    def __init__(self, envs, config):
        """
        Initialize agent, networks, optimizer, buffer.

        Args:
            envs: Vectorized gym environment.
            config: Dictionary of hyperparameters.
        """
        self.envs = envs
        self.config = config
        self.global_step = 0

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        key = jr.key(config["seed"])
        key, q_key = jr.split(key)

        obs, _ = envs.reset(seed=config["seed"])
        self.q_network = QNetwork(action_dim=envs.single_action_space.n)
        init_params = self.q_network.init(q_key, obs)

        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=init_params,
            target_params=init_params,
            tx=optax.adamw(learning_rate=config["learning_rate"])
        )
        self.q_network.apply = jax.jit(self.q_network.apply)
        self.q_state = self.q_state.replace(
            target_params=optax.incremental_update(self.q_state.params, self.q_state.target_params, 1)
        )

        self.rb = ReplayBuffer(
            config["buffer_size"],
            envs.single_observation_space,
            envs.single_action_space,
        )

        self.train_frequency = config["train_frequency"]
        self.epsilon = config["epsilon"]
        self.epsilon_final = config["epsilon_final"]
        self.exploration_fraction = config["exploration_fraction"]
        self.total_timesteps = config["total_timesteps"]
        self.tau = config["tau"]
        self.log_info = {}
        self.unique_state_ids = set()

    @staticmethod
    def linear_schedule(start_e, end_e, duration, t):
        """
        Linear annealing of epsilon.

        Args:
            start_e: Initial epsilon.
            end_e: Final epsilon.
            duration: Total annealing steps.
            t: Current step.

        Returns:
            Epsilon at step t.
        """
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def train_step(self, global_step):
        """
        One DQN update step if enough data.

        Args:
            global_step: Current step count.
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

        num_actions = self.envs.single_action_space.n
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
            "rewards/max_extrinsic": float(jnp.max(batch.rewards)),
            "exploration/episode_count": float(jnp.sum(batch.dones)),
            "exploration/unique_states": len(self.unique_state_ids),
            "steps/global": global_step,
        }

    def update_target_network(self):
        """
        Polyak update of target network.
        """
        self.q_state = self.q_state.replace(
            target_params=optax.incremental_update(
                self.q_state.params, self.q_state.target_params, self.tau
            )
        )

    def record_step(self, obs, next_obs, actions, rewards, terminations, infos, global_step):
        """
        Save transition to replay buffer.

        Args:
            obs: Current observations.
            next_obs: Next observations.
            actions: Actions taken.
            rewards: Rewards received.
            terminations: Done flags.
            infos: Info dict.
            global_step: Current step count.
        """
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(terminations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        try:
            obs_np = np.asarray(obs)
            for ob in obs_np:
                obs_hash = hashlib.sha1(ob.tobytes()).hexdigest()
                self.unique_state_ids.add(obs_hash)
        except Exception as e:
            print(f"Warning: Failed to hash obs for unique states: {e}")

        self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    def log_metrics(self, global_step, start_time):
        """
        Add steps per second to log info.

        Args:
            global_step: Current step count.
            start_time: Start time of training.

        Returns:
            Dict of logged metrics.
        """
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def select_action(self, obs, global_step):
        """
        Select action with epsilon-greedy exploration.

        Args:
            obs: Current observations.
            global_step: Current step count.

        Returns:
            Actions as numpy array.
        """
        epsilon = self.linear_schedule(
            self.epsilon, self.epsilon_final,
            self.exploration_fraction * self.total_timesteps, global_step
        )
        self.log_info.update({"exploration/epsilon": float(epsilon)})
        if random.random() < epsilon:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            q_values = self.q_network.apply(self.q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        return actions

    @partial(jax.jit, static_argnums=0)
    def update(self, q_state, observations, actions, next_observations, rewards, dones):
        """
        DQN loss and gradient update.

        Args:
            q_state: Current network state.
            observations: Current states.
            actions: Actions taken.
            next_observations: Next states.
            rewards: Rewards.
            dones: Done flags.

        Returns:
            Loss, grad norm, Q-values, TD-targets, updated state.
        """
        q_next_target = self.q_network.apply(q_state.target_params, next_observations)
        q_next_target = jnp.max(q_next_target, axis=-1)
        td_target = rewards + (1 - dones) * self.config["gamma"] * q_next_target

        def mse_loss(params):
            q_pred = self.q_network.apply(params, observations)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            return ((q_pred - td_target) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)

        squared_grad_norms = jax.tree_util.tree_map(lambda g: jnp.mean(g ** 2), grads)
        flat_grads, _ = jax.flatten_util.ravel_pytree(squared_grad_norms)
        grad_mse = jnp.mean(flat_grads)

        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, grad_mse, q_pred, td_target, q_state