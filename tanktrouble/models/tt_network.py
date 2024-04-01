from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import ActorFactory
from tianshou.highlevel.module.core import (
    TDevice,
)
from tianshou.highlevel.module.intermediate import (
    IntermediateModule,
    IntermediateModuleFactory,
)
from tianshou.utils.net.discrete import Actor, NoisyLinear


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ScaledObsInputModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, denom: float = 255.0) -> None:
        super().__init__()
        self.module = module
        self.denom = denom
        # This is required such that the value can be retrieved by downstream modules (see usages of get_output_dim)
        self.output_dim = module.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if info is None:
            info = {}
        return self.module.forward(obs / self.denom, state, info)


def scale_obs(module: nn.Module, denom: float = 255.0) -> nn.Module:
    return ScaledObsInputModule(module, denom=denom)


class DQN_TT(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        rest: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        c: int = 0,
        h: int = 0,
        w: int = 0,
        hidden: Sequence[int] = 512,
        use_img: bool = True,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
        grad: bool = True,
        features_only: bool = False,
        atoms: int = 1,
    ) -> None:
        super().__init__()
        self.use_img = use_img
        self.image_size = c*h*w
        self.rest = rest
        self.total_size = self.image_size + self.rest
        self.action_shape = action_shape
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        self.grad = grad
        self.features_only = features_only
        self.atoms = atoms
        if use_img:
            self.cnet = nn.Sequential(
                layer_init(nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(inplace=True),
                layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(inplace=True),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            )
            with torch.no_grad():
                self.output_dim = int(np.prod(self.cnet(torch.zeros(1, c, h, w)).shape[1:])) + self.rest
        else:
            self.output_dim = self.rest
            self.total_size = self.rest
        # self.anet = nn.Sequential(
        #         layer_init(nn.Linear(self.output_dim, hidden)),
        #         nn.ReLU(inplace=True),
        #     layer_init(nn.Linear(hidden, int(np.prod(action_shape)))),
        # )
        if not self.features_only:
            self.anet = nn.ModuleList()
            self.anet.append(layer_init(nn.Linear(self.output_dim, hidden[0])))
            self.anet.append(nn.ReLU(inplace=True))
            for i in range(1, len(hidden)):
                self.anet.append(layer_init(nn.Linear(hidden[i-1], hidden[i])))
                self.anet.append(nn.ReLU(inplace=True))
            self.anet.append(layer_init(nn.Linear(hidden[-1], int(np.prod(action_shape)) * atoms)))
            self.output_dim = np.prod(action_shape) * atoms

    def _forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if info is None:
            info = {}
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.reshape(-1, self.total_size)
        if self.use_img:
            conv_input = obs[:, :self.image_size].reshape(-1, self.c, self.h, self.w)
            conv_features = self.cnet(conv_input)
            rest_input = obs[:, self.image_size:]
            anet_input = torch.cat([conv_features, rest_input], dim=1)
        else:
            anet_input = obs
        if self.features_only:
            return anet_input, state
        for layer in self.anet:
            anet_input = layer(anet_input)
        if self.atoms != 1:
            anet_input = anet_input.view(-1, int(np.prod(self.action_shape)), self.atoms)
            anet_input = anet_input.softmax(dim=2)
        return anet_input, state

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        return self._forward(obs, state, info)




class Rainbow(DQN_TT):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: str | int | torch.device = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512),
            nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms),
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512),
                nn.ReLU(inplace=True),
                linear(512, self.num_atoms),
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        if info is None:
            info = {}
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state


#
# class C51(DQN):
#     """Reference: A distributional perspective on reinforcement learning.
#
#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """
#
#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_atoms: int = 51,
#         device: str | int | torch.device = "cpu",
#     ) -> None:
#         self.action_num = np.prod(action_shape)
#         super().__init__(c, h, w, [self.action_num * num_atoms], device)
#         self.num_atoms = num_atoms
#
#     def forward(
#         self,
#         obs: np.ndarray | torch.Tensor,
#         state: Any | None = None,
#         info: dict[str, Any] | None = None,
#     ) -> tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         if info is None:
#             info = {}
#         obs, state = super().forward(obs)
#         obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
#         obs = obs.view(-1, self.action_num, self.num_atoms)
#         return obs, state
#
#
# class Rainbow(DQN):
#     """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
#
#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """
#
#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_atoms: int = 51,
#         noisy_std: float = 0.5,
#         device: str | int | torch.device = "cpu",
#         is_dueling: bool = True,
#         is_noisy: bool = True,
#     ) -> None:
#         super().__init__(c, h, w, action_shape, device, features_only=True)
#         self.action_num = np.prod(action_shape)
#         self.num_atoms = num_atoms
#
#         def linear(x, y):
#             if is_noisy:
#                 return NoisyLinear(x, y, noisy_std)
#             return nn.Linear(x, y)
#
#         self.Q = nn.Sequential(
#             linear(self.output_dim, 512),
#             nn.ReLU(inplace=True),
#             linear(512, self.action_num * self.num_atoms),
#         )
#         self._is_dueling = is_dueling
#         if self._is_dueling:
#             self.V = nn.Sequential(
#                 linear(self.output_dim, 512),
#                 nn.ReLU(inplace=True),
#                 linear(512, self.num_atoms),
#             )
#         self.output_dim = self.action_num * self.num_atoms
#
#     def forward(
#         self,
#         obs: np.ndarray | torch.Tensor,
#         state: Any | None = None,
#         info: dict[str, Any] | None = None,
#     ) -> tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         if info is None:
#             info = {}
#         obs, state = super().forward(obs)
#         q = self.Q(obs)
#         q = q.view(-1, self.action_num, self.num_atoms)
#         if self._is_dueling:
#             v = self.V(obs)
#             v = v.view(-1, 1, self.num_atoms)
#             logits = q - q.mean(dim=1, keepdim=True) + v
#         else:
#             logits = q
#         probs = logits.softmax(dim=2)
#         return probs, state
#
#
# class QRDQN(DQN):
#     """Reference: Distributional Reinforcement Learning with Quantile Regression.
#
#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """
#
#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_quantiles: int = 200,
#         device: str | int | torch.device = "cpu",
#     ) -> None:
#         self.action_num = np.prod(action_shape)
#         super().__init__(c, h, w, [self.action_num * num_quantiles], device)
#         self.num_quantiles = num_quantiles
#
#     def forward(
#         self,
#         obs: np.ndarray | torch.Tensor,
#         state: Any | None = None,
#         info: dict[str, Any] | None = None,
#     ) -> tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         if info is None:
#             info = {}
#         obs, state = super().forward(obs)
#         obs = obs.view(-1, self.action_num, self.num_quantiles)
#         return obs, state
#
#
# class ActorFactoryAtariDQN(ActorFactory):
#     def __init__(
#         self,
#         hidden_size: int | Sequence[int],
#         scale_obs: bool,
#         features_only: bool,
#     ) -> None:
#         self.hidden_size = hidden_size
#         self.scale_obs = scale_obs
#         self.features_only = features_only
#
#     def create_module(self, envs: Environments, device: TDevice) -> Actor:
#         net = DQN(
#             *envs.get_observation_shape(),
#             envs.get_action_shape(),
#             device=device,
#             features_only=self.features_only,
#             output_dim=self.hidden_size,
#             layer_init=layer_init,
#         )
#         if self.scale_obs:
#             net = scale_obs(net)
#         return Actor(net, envs.get_action_shape(), device=device, softmax_output=False).to(device)
#

class IntermediateModuleFactoryAtariDQN(IntermediateModuleFactory):
    def __init__(self, features_only: bool = False, net_only: bool = False) -> None:
        self.features_only = features_only
        self.net_only = net_only

    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        dqn = DQN(
            *envs.get_observation_shape(),
            envs.get_action_shape(),
            device=device,
            features_only=self.features_only,
        ).to(device)
        module = dqn.net if self.net_only else dqn
        return IntermediateModule(module, dqn.output_dim)


class IntermediateModuleFactoryAtariDQNFeatures(IntermediateModuleFactoryAtariDQN):
    def __init__(self) -> None:
        super().__init__(features_only=True, net_only=True)
