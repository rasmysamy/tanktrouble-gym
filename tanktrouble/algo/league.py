from collections import deque

import gymnasium
import numpy as np
from tianshou.policy import BasePolicy, TrainingStats


class LeaguePolicy(BasePolicy):
    def __init__(self, policy_list=[], max_agents=10, *, action_space: gymnasium.Space):
        super().__init__(action_space=action_space)
        self.policy_list = policy_list
        self.max_agents = max_agents
        self.idx = 0

    def update_idx(self):
        self.idx = np.random.randint(0, len(self.policy_list))

    def forward(self, batch, state=dict(), **kwargs):
        if len(self.policy_list) == 0:
            raise ValueError("No policies in the queue")
        # choose a random agent from the list
        ag_idx = self.idx % len(self.policy_list)
        action = self.policy_list[ag_idx].forward(batch, state, **kwargs)
        return action

    def push_agent(self, policy, rotate=True):
        if len(self.policy_list) >= self.max_agents:
            if rotate:
                self.pop_agent()
            else:
                raise ValueError("Max agents reached")
        self.policy_list.append(policy)

    def pop_agent(self):
        if len(self.policy_list) == 0:
                raise ValueError("No agents to pop")
        self.policy_list = self.policy_list[1:]

    def learn(self, batch, **kwargs):
        return TrainingStats(train_time=0, smoothed_loss=dict(), **kwargs)

    def set_eps(self, eps):
        for policy in self.policy_list:
            policy.set_eps(eps)

