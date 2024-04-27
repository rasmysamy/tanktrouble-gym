"""This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""
from typing import Any, Literal, Protocol, Self, cast, overload

import argparse
import copy
import datetime
import os
import sys
from datetime import time
from math import prod
from time import sleep
from typing import Optional, Tuple

import gymnasium
import numpy as np
import pygame
import torch
from pettingzoo.utils import parallel_to_aec
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer, ReplayBuffer
from tianshou.data.batch import BatchProtocol, Batch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, RainbowPolicy, ICMPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

import tanktrouble.env.tanktrouble_env as tanktrouble
from tanktrouble.models.tt_network import DQN_TT, Rainbow

from league import LeaguePolicy

from pettingzoo.classic import tictactoe_v3

is_distrib = False
icm = False
import multiprocessing

s_x = 8
s_y = 5
img_size = (s_x * 3, s_y * 3, 5)

watch = False


def _get_agents(
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        league=True,
        single=False,
        watch=False,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["0"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    hidden_sizes = [512]
    if agent_learn is None:
        # model
        action_dim = gymnasium.spaces.utils.flatdim(env.action_space)

        def make_net(grad=True, features_only=False):
            # return Net(
            #     state_shape=gymnasium.spaces.utils.flatdim(observation_space),
            #     action_shape=gymnasium.spaces.utils.flatdim(env.action_space),
            #     hidden_sizes=hidden_sizes,
            #     device="cuda" if torch.cuda.is_available() else "cpu",
            #     num_atoms=51 if is_distrib else 1,
            # ).to("cuda" if torch.cuda.is_available() else "cpu")
            return DQN_TT(
                rest=gymnasium.spaces.utils.flatdim(observation_space) - (
                    prod(img_size) if tanktrouble.image_in_obs else 0),
                action_shape=[action_dim],
                hidden=hidden_sizes,
                c=img_size[2] if tanktrouble.image_in_obs else 0,
                h=img_size[1] if tanktrouble.image_in_obs else 0,
                w=img_size[0] if tanktrouble.image_in_obs else 0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_img=tanktrouble.image_in_obs,
                grad=grad,
                features_only=features_only or is_distrib,
                atoms=51 if is_distrib else 1,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

        def make_rainbow_net():
            return Rainbow(
                rest=gymnasium.spaces.utils.flatdim(observation_space) - (
                    prod(img_size) if tanktrouble.image_in_obs else 0),
                action_shape=[action_dim],
                c=img_size[2] if tanktrouble.image_in_obs else 0,
                h=img_size[1] if tanktrouble.image_in_obs else 0,
                w=img_size[0] if tanktrouble.image_in_obs else 0,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        net = make_net()
        net_fixed = make_net(grad=False).train(False)

        optim = torch.optim.Adam(net.parameters(), lr=3e-4)
        optim_fixed = torch.optim.SGD(net_fixed.parameters(), lr=0.0)
        icm_optim = torch.optim.Adam(net.parameters(), lr=3e-4)

        if not is_distrib:
            agent_learn = DQNPolicy(
                model=net,
                optim=optim,
                # discount_factor=0.98,
                estimation_step=5,
                target_update_freq=1500,
                action_space=env.action_space,
                is_double=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            agent_fixed = DQNPolicy(
                model=net_fixed,
                optim=optim_fixed,
                # discount_factor=0.98,
                estimation_step=5,
                target_update_freq=1500,
                action_space=env.action_space,
                is_double=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        if icm and not is_distrib:
            feature_dim = net.output_dim
            feature_net = make_net(features_only=True)
            feature_net_fixed = make_net(features_only=True).train(False)
            icm_net = IntrinsicCuriosityModule(
                feature_net=net,
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            icm_net_fixed = IntrinsicCuriosityModule(
                feature_net=net_fixed,
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            icm_lr_scale = 1.0
            icm_reward_scale = 0.01
            icm_forward_loss_weight = 0.2
            agent_learn = ICMPolicy(
                policy=agent_learn,
                model=icm_net,
                optim=icm_optim,
                action_space=env.action_space,
                lr_scale=icm_lr_scale,
                reward_scale=icm_reward_scale,
                forward_loss_weight=icm_forward_loss_weight,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            agent_fixed = ICMPolicy(
                policy=agent_fixed,
                model=icm_net_fixed,
                optim=optim_fixed,
                action_space=env.action_space,
                lr_scale=icm_lr_scale,
                reward_scale=icm_reward_scale,
                forward_loss_weight=icm_forward_loss_weight,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

        if is_distrib:
            rnet = make_rainbow_net()
            rnet_fixed = make_rainbow_net()
            agent_learn = RainbowPolicy(model=rnet, optim=optim, action_space=env.action_space, estimation_step=4,
                                        target_update_freq=1000).to("cuda" if torch.cuda.is_available() else "cpu")
            agent_fixed = RainbowPolicy(model=rnet_fixed, optim=optim_fixed, action_space=env.action_space,
                                        estimation_step=4, target_update_freq=1000).to(
                "cuda" if torch.cuda.is_available() else "cpu")

    if single:
        return agent_fixed

    from tianshou.data import Batch

    def do_nothing(
            self,
            batch: Batch,
            state=None,
            **kwargs,
    ) -> Batch:
        mask = batch.obs.mask
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        return Batch(act=np.zeros_like(logits.argmax(axis=-1)))

    def _nothing_set_eps(self, eps):
        self.eps = eps

    def _nothing_exploration_noise(self, act, batch):
        return np.where(np.random.rand(*act.shape) < self.eps, np.random.randint(0, act.shape[-1], size=act.shape[0]), act)

    train_agent_idx = "1"

    do_nothing_agent = RandomPolicy(action_space=env.action_space)
    do_nothing_agent.forward = do_nothing.__get__(do_nothing_agent)
    do_nothing_agent.set_eps = _nothing_set_eps.__get__(do_nothing_agent)
    do_nothing_agent.exploration_noise = _nothing_exploration_noise.__get__(do_nothing_agent)
    do_nothing_agent.set_eps(0)

    def multi_post_process(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        # concat all weights:
        for k, val in batch.items():
            agent_index = np.nonzero(batch.obs.agent_id == k)[0]
            if k == train_agent_idx:
                self.policies[k].post_process_fn(batch=val, buffer=buffer, indices=indices[agent_index])

    # agents = [LeaguePolicy([agent_fixed], max_agents=10, action_space=env.action_space) if league else agent_fixed,
    #           agent_learn]
    # agents = [RandomPolicy(action_space=env.action_space), agent_learn]
    # agents = [do_nothing_agent, agent_learn]
    agents = [agent_fixed, agent_learn]
    policy = MultiAgentPolicyManager(policies=agents, env=env)
    return policy, optim, env.agents


def _get_env():
    env = tanktrouble.TankTrouble()
    env.reset()
    env.set_onehot(True)
    return PettingZooEnv(parallel_to_aec(env))


def watch_self_play(user_play=False):
    if user_play:
        from pynput import keyboard
        pressed = {'w': False, 's': False, 'a': False, 'd': False, 'up': False, 'down': False, 'left': False,
                   'right': False,
                   'fire': False, 'enter': False, 'esc': False}


        def on_press(key):
            if isinstance(key, keyboard.Key):
                if key is keyboard.Key.up:
                    pressed['up'] = True
                elif key is keyboard.Key.down:
                    pressed['down'] = True
                elif key is keyboard.Key.left:
                    pressed['left'] = True
                elif key is keyboard.Key.right:
                    pressed['right'] = True
                return
            print(key.char, "pressed")
            pressed[key.char] = True

        def on_release(key):
            if isinstance(key, keyboard.Key):
                if key is keyboard.Key.space:
                    pressed['fire'] = True
                elif key is keyboard.Key.enter:
                    pressed['enter'] = True
                elif key is keyboard.Key.up:
                    pressed['up'] = False
                elif key is keyboard.Key.down:
                    pressed['down'] = False
                elif key is keyboard.Key.left:
                    pressed['left'] = False
                elif key is keyboard.Key.right:
                    pressed['right'] = False
                elif key is keyboard.Key.esc:
                    pressed['esc'] = True
                return
            print(key.char, "released")
            pressed[key.char] = False

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()


    env = tanktrouble.TankTrouble()
    env.reset()
    env.set_onehot(True)
    policy, optim, ag = _get_agents(league=False)
    policy.policies[ag[1]].load_state_dict(torch.load("log/ttt/dqn/policy.pth"))
    # policy.policies[ag[0]].load_state_dict(policy.policies[ag[1]].state_dict().copy())
    policy.policies[ag[1]].set_eps(0.001)
    policy.policies[ag[0]].set_eps(0.001)
    if os.path.islink("log/ttt/dqn/policy.pth"):
        link_target = os.readlink("log/ttt/dqn/policy.pth")
    else:
        link_target = None
    while True:
        obs, _ = env.reset()
        p1_obs = obs["0"]["observation"]
        p2_obs = obs["1"]["observation"]
        # act1 = policy.policies[ag[0]].compute_action(p1_obs)
        act1 = 0
        act2 = policy.policies[ag[1]].compute_action(p2_obs)
        while True:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            sleep(1 / 60.0)
            # print(act2)
            obs, rews, done, _, _ = env.step({"0": act1, "1": act2})
            p1_obs = obs["0"]["observation"]
            p2_obs = obs["1"]["observation"]
            b1 = Batch(obs=p1_obs, act=act1, rew=rews["0"], done=done["0"])
            b2 = Batch(obs=p2_obs, act=act2, rew=rews["1"], done=done["1"])
            if not user_play:
                act1 = policy.policies[ag[0]].compute_action(p1_obs)
                act1 = policy.policies[ag[0]].exploration_noise(np.array([act1]), b1)[0]
            else:
                act1 = [pressed['w'], pressed['s'], pressed['a'], pressed['d'], pressed['fire']]
                act1 = env.action_to_onehot(act1)
                pressed['fire'] = False
            act2 = policy.policies[ag[1]].compute_action(p2_obs)
            act2 = policy.policies[ag[1]].exploration_noise(np.array([act2]), b2)[0]
            if done["0"]:
                if link_target is not None:
                    if link_target != os.readlink("log/ttt/dqn/policy.pth"):
                        return
                break


if __name__ == "__main__":
    # get commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Watch the trained agent play")
    args = parser.parse_args()
    if args.watch:
        while True:
            watch_self_play()


    # ======== Step 1: Environment setup =========
    def with_render(env):
        env.always_render = True
        return env

    env = _get_env()


    train_envs = SubprocVectorEnv([_get_env for _ in range(8)])
    test_envs = SubprocVectorEnv([_get_env for _ in range(8)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents(league=False)

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        # HERVectorReplayBuffer(20_000, len(train_envs), compute_reward_fn=lambda x: x[1], horizon=1000),
        VectorReplayBuffer(1_000_000, len(train_envs)),
        # PrioritizedVectorReplayBuffer(1_000_000, len(train_envs), alpha=0.6, beta=0.4, weight_norm=True, ignore_obs_next=True),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num


    def update_league_and_forward(self, fun, *args, **kwargs):
        for pol in self.policy.policies.values():
            if hasattr(pol, "update_idx"):
                pol.update_idx()
        fun(*args, **kwargs)


    # test_collector.reset_env__ = test_collector.reset_env
    # test_collector._reset_env_with_ids__ = test_collector._reset_env_with_ids
    # test_collector.reset_env = lambda *args, **kwargs: update_league_and_forward(test_collector, test_collector.reset_env__, *args, **kwargs)
    # test_collector._reset_env_with_ids = lambda *args, **kwargs: update_league_and_forward(test_collector, test_collector._reset_env_with_ids__, *args, **kwargs)

    # train_collector.reset_env__ = train_collector.reset_env
    # train_collector._reset_env_with_ids__ = train_collector._reset_env_with_ids
    # train_collector.reset_env = lambda *args, **kwargs: update_league_and_forward(train_collector, train_collector.reset_env__, *args, **kwargs)
    # train_collector._reset_env_with_ids = lambda *args, **kwargs: update_league_and_forward(test_collector, train_collector._reset_env_with_ids__, *args, **kwargs)

    def override_kwargs_and_forward(fun, *args, **kwargs):
        kwargs["render"] = 0.00000001
        return fun(*args, **kwargs)


    # test_collector.collect___b = test_collector.collect
    # test_collector.collect___r = lambda *args, **kwargs: override_kwargs_and_forward(test_collector.collect___b, *args, **kwargs)
    # test_collector.collect = test_collector.collect___r

    thresh_win_rate = 0.6
    thresh_in_a_row = 4
    in_a_row = 0
    wins = 0
    losses = 0
    draws = 0


    max_opps = 10
    opponent_list = [_get_agents(single=True) for _ in range(max_opps)]

    # ======== Step 4: Callback functions setup =========
    if watch:
        # multiprocessing.set_start_method("spawn")
        p = multiprocessing.Process(target=watch_self_play)


    def save_best_fn(policy):
        global p
        # get datetime
        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_save_path = os.path.join(os.getcwd(), "log", "ttt", "dqn", f"policy-{timestr}.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)
        # make symlink
        os.remove(os.path.join(os.getcwd(), "log", "ttt", "dqn", "policy.pth"))
        os.symlink(model_save_path, os.path.join(os.getcwd(), "log", "ttt", "dqn", "policy.pth"))
        # create new process to watch the trained agent play
        if watch:
            while p.is_alive():
                p.terminate()
            p = multiprocessing.Process(target=watch_self_play)
            p.start()


    def stop_fn(mean_rewards):
        global wins, losses, draws, in_a_row
        winlossrate = wins / (wins + losses + 1e-6)
        winrate = wins / (wins + draws + losses)
        print(f"win/loss rate: {winlossrate}, win rate: {winrate}, wins: {wins}, losses: {losses}, draws: {draws}")
        wins = 0
        losses = 0
        draws = 0

        if winlossrate > thresh_win_rate and mean_rewards >= 0.2:
            in_a_row += 1
        else:
            in_a_row = 0
        return in_a_row >= thresh_in_a_row


    def train_fn(epoch, env_step):
        # choose random opponent
        # opp_idx = np.random.randint(0, max_opps)
        # policy.replace_policy(opponent_list[opp_idx], 0)
        # policy.policies[agents[0]].load_state_dict(opponent_list[opp_idx].state_dict().copy())
        global wins, draws, losses
        wins = 0
        losses = 0
        draws = 0
        eps = 0.3
        if env_step > 50_000:
            eps -= 0.1 * (env_step - 50_000)/70_000
            eps = max(0.003, eps)
        policy.policies[agents[1]].set_eps(eps)
        # policy.policies[agents[0]].update_idx()
        # policy.policies[agents[0]].set_eps(eps)


    def test_fn(epoch, env_step):
        global wins, draws, losses
        wins = 0
        losses = 0
        draws = 0
        policy.policies[agents[1]].set_eps(0.001)
        policy.policies[agents[0]].set_eps(0.001)


    def reward_metric(rews):
        global wins, losses, draws
        wins += np.count_nonzero(rews[:, 1] > rews[:, 0])
        losses += np.count_nonzero(rews[:, 1] < rews[:, 0])
        draws += np.count_nonzero(rews[:, 1] == rews[:, 0])
        return rews[:, 1]





    # ======== Step 5: Run the trainer =========


    gen_count = 10_000
    for i in range(gen_count):
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=5000,
            step_per_epoch=150_000,
            step_per_collect=10,
            episode_per_test=40,
            batch_size=32,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=.1,
            test_in_train=False,
            reward_metric=reward_metric,
        ).run()
        wins = 0
        losses = 0
        draws = 0
        in_a_row = 0
        # agent_copy = copy.copy(policy.policies[agents[1]])
        # agent_copy.load_state_dict(policy.policies[agents[1]].state_dict().copy())
        # policy.policies[agents[0]].push_agent(agent_copy, rotate=True)

        print(f"Generation {i + 1}/{gen_count} complete")
        # policy.policies[agents[0]].load_state_dict(policy.policies[agents[1]].state_dict().copy())
        opponent_list = [_get_agents(single=True).load_state_dict(policy.policies[agents[1]].state_dict().copy())] + opponent_list[:-1]

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
