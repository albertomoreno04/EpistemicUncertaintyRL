from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

class RNDNetwork(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.hidden_dim)(x)

class RND:
    def __init__(self, obs_shape, config):
        self.predictor = RNDNetwork(config["predictor_hidden_dim"])
        self.target = RNDNetwork(config["target_hidden_dim"])
        dummy_input = jnp.zeros((1,) + obs_shape, dtype=jnp.float32)
        self.predictor_params = self.predictor.init(jax.random.PRNGKey(0), dummy_input)
        self.target_params = self.target.init(jax.random.PRNGKey(1), dummy_input)
        self.optimizer = optax.adam(config["learning_rate"])
        self.train_state = train_state.TrainState.create(apply_fn=self.predictor.apply,
                                                         params=self.predictor_params,
                                                         tx=self.optimizer)
        self.obs_running_mean = jnp.zeros(obs_shape)
        self.obs_running_var = jnp.ones(obs_shape)
        self.count = 1e-4

    def update_obs_stats(self, obs: jnp.ndarray):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)

        total_count = self.count + obs.shape[0]

        delta = batch_mean - self.obs_running_mean
        new_mean = self.obs_running_mean + delta * obs.shape[0] / total_count
        m_a = self.obs_running_var * self.count
        m_b = batch_var * obs.shape[0]
        M2 = m_a + m_b + jnp.square(delta) * self.count * obs.shape[0] / total_count
        new_var = M2 / total_count

        self.obs_running_mean = new_mean
        self.obs_running_var = new_var
        self.count = total_count

    @partial(jax.jit, static_argnums=0)
    def compute_intrinsic_reward(self, obs):
        obs = obs.reshape((obs.shape[0], -1))

        pred = self.predictor.apply(self.train_state.params, obs)
        target = self.target.apply(self.target_params, obs)
        reward = jnp.mean(jnp.square(pred - target), axis=-1)
        return reward

    @partial(jax.jit, static_argnums=0)
    def update(self, obs):
        obs = obs.reshape((obs.shape[0], -1))
        def loss_fn(params):
            pred = self.predictor.apply(params, obs)
            target = self.target.apply(self.target_params, obs)
            loss = jnp.mean(jnp.square(pred - target))
            return loss

        loss = loss_fn(self.train_state.params)
        grads = jax.grad(loss_fn)(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)
        return loss
