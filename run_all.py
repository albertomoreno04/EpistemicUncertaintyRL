import os
import subprocess

# Define environments and algorithms
envs = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]
agents = ["epsilon_greedy", "rnd"]

# Base command
base_command = "python main.py"

# Loop over everything
for env in envs:
    for agent in agents:
        print(f"Running {agent} on {env}...")
        command = f"{base_command} --env_id {env} --agent_type {agent}"
        subprocess.run(command, shell=True)