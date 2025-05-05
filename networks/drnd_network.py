import flax.linen as nn
import jax.numpy as jnp

class DRNDNetwork(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))

        # Output two heads: mean and log_std
        mean = nn.Dense(self.hidden_dim)(x)
        log_std = nn.Dense(self.hidden_dim)(x)

        # We apply softplus to make sure std is positive
        std = nn.softplus(log_std)

        return mean, std