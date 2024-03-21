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
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, IQNPolicy, RainbowPolicy

from tanktrouble.env import tanktrouble_env as tanktrouble
from supersuit import flatten_v0

from pettingzoo.classic import rps_v2

if __name__ != "__main__":
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
    result = collector.collect(n_episode=200, render=0.000001)


"""This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from pettingzoo.classic import tictactoe_v3


def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["0"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=gymnasium.spaces.utils.flatdim(observation_space),
            action_shape=(gymnasium.spaces.utils.flatdim(env.action_space)),
            hidden_sizes=[256,64,64,64],
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_atoms=51,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        # agent_learn = DQNPolicy(
        #     model=net,
        #     optim=optim,
        #     discount_factor=0.98,
        #     estimation_step=3,
        #     target_update_freq=3200,
        # )
        agent_learn = RainbowPolicy(model=net, optim=optim).to("cuda" if torch.cuda.is_available() else "cpu")

    if agent_opponent is None:
        agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = tanktrouble.TankTrouble()
    env.reset()
    env.set_onehot(True)
    return PettingZooEnv(parallel_to_aec(env))


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    def with_render(env):
        env.always_render = True
        return env
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    def override_kwargs_and_forward(fun, *args, **kwargs):
        kwargs["render"] = 0.00000001
        return fun(*args, **kwargs)

    test_collector.collect___b = test_collector.collect
    test_collector.collect___r = lambda *args, **kwargs: override_kwargs_and_forward(test_collector.collect___b, *args, **kwargs)
    test_collector.collect = test_collector.collect___r

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.03)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.00)
        if epoch%1 == 0:
            test_collector.collect = test_collector.collect___r
        else:
            test_collector.collect = test_collector.collect___b

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50000,
        step_per_epoch=1000,
        step_per_collect=1000,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=.2,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")