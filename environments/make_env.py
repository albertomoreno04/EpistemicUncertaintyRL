import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordEpisodeStatistics, RecordVideo, NormalizeObservation


class AddStateToInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["state"] = tuple(obs)
        return obs, reward, terminated, truncated, info

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ConvertToFloat32(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)

def make_env(env_id, seed, idx, capture_video):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # We'll manually control when videos are recorded
            # env = gym.wrappers.RecordVideo(
            #     env, f"videos/{run_name}", step_trigger=lambda s: s % 50000 == 49999
            # )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = AddStateToInfoWrapper(env)
        env.action_space.seed(seed)

        return env

    return thunk

def make_atari_env(env_id, seed, idx, capture_video, frame_stack=4, resize_shape=84):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=resize_shape)
        env = ConvertToFloat32(env)
        env = NormalizeObservation(env)
        env = ActionRepeat(env, repeat=4)
        env = FrameStack(env, num_stack=frame_stack)
        env = RecordEpisodeStatistics(env)
        env = AddStateToInfoWrapper(env)
        env.action_space.seed(seed)

        return env

    return thunk