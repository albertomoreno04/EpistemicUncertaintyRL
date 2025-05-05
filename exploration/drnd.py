from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from networks.drnd_network import DRNDNetwork


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

    def compute_intrinsic_reward(self, obs):
        mean_pred, std_pred = self.predictor.apply(self.train_state.params, obs)
        mean_target, std_target = self.target.apply(self.target_params, obs)

        # Compute Mahalanobis distance
        diff = mean_pred - mean_target
        intrinsic_reward = jnp.sum((diff ** 2) / (std_pred ** 2 + 1e-8), axis=-1)

        return intrinsic_reward

    @partial(jax.jit, static_argnums=0)
    def update_step(self, train_state, obs):
        def loss_fn(params):
            mean_pred, std_pred = self.predictor.apply(params, obs)
            mean_target, std_target = self.target.apply(self.target_params, obs)

            diff = mean_pred - mean_target
            loss = jnp.mean(jnp.sum((diff ** 2) / (std_pred ** 2 + 1e-8) + jnp.log(std_pred ** 2 + 1e-8), axis=-1))
            return loss

        grads = jax.grad(loss_fn)(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grads)
        return new_train_state

    def update(self, obs):
        # Now use the pure jitted update
        self.train_state = self.update_step(self.train_state, obs)
