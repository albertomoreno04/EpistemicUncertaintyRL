import gymnasium as gym
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)


class AddStateToInfoWrapper(gym.Wrapper):
    """
    Adds the raw state (observation) to the 'info' dict after each step.
    Useful for logging or hashing visited states.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["state"] = tuple(obs)
        return obs, reward, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video):
    """
    Creates a standard Gymnasium environment with optional video recording and state tracking.

    Args:
        env_id (str): Environment ID (e.g., 'CartPole-v1').
        seed (int): Seed for action space reproducibility.
        idx (int): Index of the environment (for multi-env setups).
        capture_video (bool): Whether to record videos for env 0.

    Returns:
        function: A thunk that returns the wrapped environment when called.
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = RecordEpisodeStatistics(env)
        env = AddStateToInfoWrapper(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_atari_env(env_id, seed, idx, capture_video, frame_stack=1, resize_shape=84):
    """
    Creates a pre-processed Atari environment with grayscale, resizing, frame stacking, and optional video.

    Args:
        env_id (str): Atari environment ID (e.g., 'ALE/Breakout-v5').
        seed (int): Seed for action space reproducibility.
        idx (int): Index of the environment (for multi-env setups).
        capture_video (bool): Whether to record videos for env 0.
        frame_stack (int): Number of frames to stack (default 1).
        resize_shape (int): Resize shape for observations (default 84).

    Returns:
        function: A thunk that returns the wrapped Atari environment when called.
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=resize_shape)
        env = FrameStack(env, num_stack=frame_stack)
        env = RecordEpisodeStatistics(env)
        env = AddStateToInfoWrapper(env)
        env.action_space.seed(seed)
        return env

    return thunk