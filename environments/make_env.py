import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordEpisodeStatistics, RecordVideo


class AddStateToInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["state"] = tuple(obs)
        return obs, reward, terminated, truncated, info

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

def make_atari_env(env_id, seed, idx, capture_video, frame_stack=1, resize_shape=84):
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