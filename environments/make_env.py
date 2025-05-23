import gymnasium as gym


class AddStateToInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["state"] = tuple(obs)
        return obs, reward, terminated, truncated, info

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # We'll manually control when videos are recorded
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", step_trigger=lambda s: s % 50000 == 49999
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = AddStateToInfoWrapper(env)
        env.action_space.seed(seed)

        return env

    return thunk