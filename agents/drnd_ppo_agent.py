from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training.train_state import TrainState
import time
import os

from networks.drnd_ppo_network import DRNDActorCritic
from exploration.drnd_ppo import DRNDPredictor, DRNDModule
from utils.replay_buffer import ReplayBuffer

class DRNDPPOAgent:
    def __init__(self, envs, config):
        self.envs = envs
        self.config = config
        self.obs_shape = envs.single_observation_space.shape
        self.action_dim = envs.single_action_space.n

        # Random key
        self.key = jax.random.PRNGKey(config["seed"])

        # Networks
        self.actor_critic = DRNDActorCritic(self.obs_shape, self.action_dim)
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        init_vars = self.actor_critic.init(self.key, dummy_obs)

        self.policy_state = TrainState.create(
            apply_fn=self.actor_critic.apply,
            params=init_vars["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["learning_rate"]),
            ),
        )

        self.drnd = DRNDModule(obs_shape=self.obs_shape, config=config)

        # Replay Buffer
        self.rb = ReplayBuffer(
            buffer_size=config["buffer_size"],
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
        )

        self.gamma = config["gamma"]
        self.lam = config["gae_lambda"]
        self.ent_coef = config["ent_coef"]
        self.clip_eps = config["clip_eps"]
        self.update_epochs = config["update_epochs"]
        self.batch_size = config["batch_size"]
        self.update_proportion = config["update_proportion"]

        self.log_info = {}

    @partial(jax.jit, static_argnums=0)
    def _select_action_jit(self, params, obs, key):
        import chex
        chex.assert_max_traces(1)
        policy_logits, value_ext, value_int = self.actor_critic.apply({"params": params}, obs)
        probs = jax.nn.softmax(policy_logits)
        action = jax.random.categorical(key, policy_logits, axis=-1)
        return action, value_ext.squeeze(), value_int.squeeze(), probs

    def select_action(self, obs, global_step):
        self.key, subkey = jax.random.split(self.key)
        action, value_ext, value_int, probs = self._select_action_jit(self.policy_state.params, obs, subkey)
        action = jax.device_get(action)

        if self.envs.num_envs == 1:
            # If only one environment, return int scalar
            action = int(action)
        else:
            action = np.asarray(action)

        return action, value_ext, value_int, probs

    def record_step(self, obs, next_obs, actions, rewards, dones, infos, global_timestep, max_steps):
        intrinsic_reward = self.drnd.compute_intrinsic_reward(next_obs)
        total_reward = rewards + self.config["intrinsic_reward_scale"] * intrinsic_reward
        self.rb.add(obs, next_obs, actions, total_reward, dones, infos)

    def log_metrics(self, global_step, start_time):
        sps = int(global_step / (time.time() - start_time))
        self.log_info["charts/SPS"] = sps
        return self.log_info

    def update_target_network(self):
        # PPO doesn't have a target network
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "policy_params": self.policy_state.params,
            "drnd_predictor_params": self.drnd.train_state.params,
        }
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(state))

    def train_step(self, global_step):
        if not self.rb.can_sample(self.batch_size):
            return

        batch = self.rb.sample(self.batch_size)
        obs, next_obs, actions, rewards, dones = (
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards.squeeze(),
            batch.dones.squeeze(),
        )

        self.policy_state, policy_loss = self._ppo_update(obs, actions, rewards, dones)

        # Update DRND predictor
        self.drnd.update(next_obs)

        self.log_info = {
            "losses/policy_loss": float(policy_loss),
        }

    @partial(jax.jit, static_argnums=0)
    def _ppo_update(self, obs, actions, rewards, dones):
        import chex
        chex.assert_max_traces(1)
        def loss_fn(params):
            logits, value_ext, value_int = self.actor_critic.apply({"params": params}, obs)
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = jnp.take_along_axis(log_probs, actions, axis=1).squeeze()

            entropy_loss = -jnp.mean(jax.scipy.special.entr(jax.nn.softmax(logits)))
            value_loss = jnp.mean((value_ext.squeeze() - rewards) ** 2)

            actor_loss = -jnp.mean(action_log_probs * rewards)

            total_loss = actor_loss + 0.5 * value_loss - self.ent_coef * entropy_loss
            return total_loss

        loss, grads = jax.value_and_grad(loss_fn)(self.policy_state.params)
        updates, new_opt_state = self.policy_state.tx.update(grads, self.policy_state.opt_state)
        new_params = optax.apply_updates(self.policy_state.params, updates)
        new_policy_state = self.policy_state.replace(
            step=self.policy_state.step + 1,
            params=new_params,
            opt_state=new_opt_state
        )
        return new_policy_state, loss
