from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from networks.drnd_network import DRNDNetwork
import chex


class DRND:
    def __init__(self, obs_shape, config):
        self.predictor = DRNDNetwork(config["predictor_hidden_dim"])
        self.target = DRNDNetwork(config["target_hidden_dim"])

        dummy_input = jnp.zeros((1, *obs_shape))

        self.predictor_params = self.predictor.init(jax.random.PRNGKey(0), dummy_input)
        self.target_params = self.target.init(jax.random.PRNGKey(1), dummy_input)

        self.optimizer = optax.adam(config["learning_rate"])
        self.train_state = train_state.TrainState.create(
            apply_fn=self.predictor.apply,
            params=self.predictor_params,
            tx=self.optimizer
        )

    def compute_intrinsic_reward(self, obs, global_step, max_steps):
        mean_pred, std_pred = self.predictor.apply(self.train_state.params, obs)
        mean_target, std_target = self.target.apply(self.target_params, obs)

        b1 = jnp.sum((mean_pred - mean_target) ** 2 / (std_pred ** 2 + 1e-8), axis=-1)

        b2 = jnp.sum((std_pred ** 2 - std_target ** 2) / (std_pred ** 2 + 1e-8), axis=-1)

        alpha = self.dynamic_alpha(global_step, max_steps)  # TODO: Could make it dynamic
        intrinsic_reward = alpha * b1 + (1 - alpha) * b2

        return intrinsic_reward

    def dynamic_alpha(self, global_step, max_steps):
        # Decrease alpha from 1 to 0 as training progresses
        return max(0, 1 - global_step / max_steps)

    @partial(jax.jit, static_argnums=0)
    def update_step(self, train_state, obs):
        def loss_fn(params):
            mean_pred, std_pred = self.predictor.apply(params, obs)
            mean_target, std_target = self.target.apply(self.target_params, obs)

            diff = mean_pred - mean_target
            loss = jnp.mean(jnp.sum((diff ** 2) / (std_pred ** 2 + 1e-8) + jnp.log(std_pred ** 2 + 1e-8), axis=-1))
            return loss

        chex.assert_max_traces(1)
        grads = jax.grad(loss_fn)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        return new_train_state

    def update(self, obs):
        # Now use the pure jitted update
        self.train_state = self.update_step(self.train_state, obs)
