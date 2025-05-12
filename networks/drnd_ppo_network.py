import flax.linen as nn

class DRNDActorCritic(nn.Module):
    obs_shape: tuple
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Flatten if necessary
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))

        # Shared hidden layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim)(x)

        # Critic heads
        value_ext = nn.Dense(1)(x)
        value_int = nn.Dense(1)(x)

        return logits, value_ext, value_int
