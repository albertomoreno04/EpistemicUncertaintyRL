import gymnasium as gym
import gymnax
from gymnax.environments.spaces import Box as GxBox, Discrete as GxDiscrete
from gymnasium.spaces import Box, Discrete
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap


def convert_space(space):
    if isinstance(space, GxBox):
        return Box(low=float(space.low), high=float(space.high), shape=space.shape, dtype=np.float32)
    elif isinstance(space, GxDiscrete):
        return Discrete(space.n)
    else:
        raise NotImplementedError(f"Unsupported space: {type(space)}")

class GymnaxWrapper:
    def __init__(self, env_id, key, key_reset, key_step, seed=0):

        self.key = key
        self.key_reset = key_reset
        self.key_step = key_step
        self.states = None
        self.num_envs = 1

        obs_space = self.env.observation_space(self.env_params)
        act_space = self.env.action_space(self.env_params)
        self.single_observation_space = convert_space(obs_space)
        self.single_action_space = convert_space(act_space)

    def reset(self):
        keys = jax.random.split(self.key_reset, self.num_envs + 1)
        self.key_step, reset_keys = keys[0], keys[1:]

        def single_reset(key):
            return self.env.reset(key, self.env_params)

        obs, states = vmap(single_reset)(reset_keys)
        self.states = states
        obs = obs.reshape(self.num_envs, -1)
        return obs, states

    def step(self, actions):
        keys = jax.random.split(self.key_step, self.num_envs + 1)
        self.key_step, step_keys = keys[0], keys[1:]

        def single_step(key, state, action):
            return self.env.step(key, state, action, self.env_params)

        obs, new_states, rewards, dones, infos = vmap(single_step)(step_keys, self.states, actions)
        self.states = new_states
        truncations = jnp.zeros_like(dones, dtype=bool)

        obs = obs.reshape(self.num_envs, -1)

        return obs, rewards, dones, truncations, infos

    def close(self):
        pass