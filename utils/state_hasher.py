import jax
import jax.numpy as jnp
from functools import partial

class StateHasher:
    def __init__(self, obs_dim, hash_size=32, seed=0):
        key = jax.random.PRNGKey(seed)
        # Random binary projection matrix for hashing
        self.proj = jax.random.normal(key, (hash_size, obs_dim))

    @partial(jax.jit, static_argnums=0)
    def hash_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        obs: (batch_size, obs_dim)
        Returns: (batch_size,) int32 hash per observation
        """
        flat_obs = obs.reshape((obs.shape[0], -1))

        projected = jnp.dot(flat_obs, self.proj.T)
        bits = (projected > 0).astype(jnp.uint32)

        powers = 2 ** jnp.arange(bits.shape[-1], dtype=jnp.uint32)
        return jnp.dot(bits, powers)