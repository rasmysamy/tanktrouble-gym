"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
from matplotlib import pyplot as plt
from pettingzoo.utils import parallel_to_aec
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from tanktrouble.env import tanktrouble_env as tanktrouble
from supersuit import flatten_v0

from pettingzoo.classic import rps_v2

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = tanktrouble.TankTrouble()
    env.reset()
    env.set_onehot(True)

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(parallel_to_aec(env))

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 second
    result = collector.collect(n_episode=200, render=0.01)