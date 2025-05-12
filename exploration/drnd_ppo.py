import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

class DRNDPredictor(nn.Module):
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x):
        # Flatten if necessary
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return x

class DRNDTarget(nn.Module):
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        return x

class DRNDModule:
    def __init__(self, obs_shape, config):
        self.predictor = DRNDPredictor(hidden_dim=config["predictor_hidden_dim"])
        self.target_networks = [
            DRNDTarget(hidden_dim=config["target_hidden_dim"])
            for _ in range(config["num_target_networks"])
        ]

        dummy_input = jnp.zeros((1, *obs_shape))

        self.predictor_params = self.predictor.init(jax.random.PRNGKey(0), dummy_input)
        self.target_params = [t.init(jax.random.PRNGKey(i+1), dummy_input) for i, t in enumerate(self.target_networks)]

        self.optimizer = optax.adam(config["learning_rate"])
        self.train_state = train_state.TrainState.create(
            apply_fn=self.predictor.apply,
            params=self.predictor_params,
            tx=self.optimizer,
        )
        self.config = config

    def compute_intrinsic_reward(self, obs):
        predict_feature = self.predictor.apply(self.train_state.params, obs)

        # Stack all target outputs
        target_features = jnp.stack([t.apply(params, obs) for t, params in zip(self.target_networks, self.target_params)], axis=0)

        # Calculate the Mahalanobis-like distance
        mu = jnp.mean(target_features, axis=0)
        B2 = jnp.mean(target_features ** 2, axis=0)

        alpha = self.config.get("alpha", 0.95)

        intrinsic_reward = alpha * jnp.sum((predict_feature - mu) ** 2, axis=-1) \
                         + (1 - alpha) * jnp.sum(jnp.sqrt(jnp.clip(jnp.abs(predict_feature ** 2 - mu ** 2) / (B2 - mu ** 2 + 1e-8), 1e-6, 1)), axis=-1)
        return intrinsic_reward


    def update(self, obs):
        def loss_fn(params):
            predict_feature = self.predictor.apply(params, obs)

            # Randomly pick target networks for each sample
            idx = jax.random.randint(jax.random.PRNGKey(0), shape=(obs.shape[0],), minval=0, maxval=len(self.target_networks))
            selected_targets = jnp.stack([self.target_networks[i].apply(self.target_params[i], obs[i:i+1])[0] for i in idx])

            # Forward loss (mean squared error)
            loss = jnp.mean((predict_feature - selected_targets) ** 2)
            return loss

        grads = jax.grad(loss_fn)(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)
