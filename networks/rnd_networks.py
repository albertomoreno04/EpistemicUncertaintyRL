import flax.linen as nn

class RNDNetwork(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.hidden_dim)(x)

class PolicyModel(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(self.action_dim)(x)
