# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce
from typing import Any, NoReturn
from tensordict import TensorDict

from rsl_rl.utils import get_param, resolve_nn_activation
from rsl_rl.modules import EmpiricalNormalization


class MLP(nn.Sequential):
    """Multi-layer perceptron with optional Layer Normalization.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        layer_norm: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
            layer_norm: Whether to apply layer normalization after each linear layer (except the last one).
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x


class ActorCriticLayerNorm(nn.Module):
    """Actor-Critic with Layer Normalization support."""
    
    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_layer_norm: bool = False,
        critic_layer_norm: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        # Pass kwargs to parent to avoid erroring on unknown args,
        # but we also need to manually initialize everything because parent uses rsl_rl MLP

        # We cannot simply call super().__init__ because it hardcodes rsl_rl.networks.MLP
        # So we duplicate the init logic here but use our custom MLP.

        nn.Module.__init__(self) # Skip ActorCritic.__init__
        self.is_recurrent = False

        if kwargs:
           print("ActorCriticLayerNorm got unexpected arguments: " + str([key for key in kwargs]))

        assert len(obs["actor"].shape) == 2, "The ActorCritic module only supports 1D observations."
        assert len(obs["critic"].shape) == 2, "The ActorCritic module only supports 1D observations."
        num_actor_obs = obs["actor"].shape[-1]
        num_critic_obs = obs["critic"].shape[-1]

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation, layer_norm=actor_layer_norm)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation, layer_norm=actor_layer_norm)
        print(f"Actor MLP (LN={actor_layer_norm}): {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation, layer_norm=critic_layer_norm)
        print(f"Critic MLP (LN={critic_layer_norm}): {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            # Finding the last linear layer in the sequential model is tricky with LN injected
            # But MLP logic puts linear layer as last or second to last.
            # However, MLP class appends layers sequentially.
            # Let's find the last Linear layer.
            last_linear = None
            for module in self.actor:
                if isinstance(module, nn.Linear):
                    last_linear = module
            
            if last_linear is not None:
                torch.nn.init.zeros_(last_linear.weight[num_actions:])
                if self.noise_std_type == "scalar":
                    torch.nn.init.constant_(last_linear.bias[num_actions:], init_noise_std)
                elif self.noise_std_type == "log":
                    torch.nn.init.constant_(
                        last_linear.bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                    )
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        # Action distribution
        self.distribution = None

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return obs["actor"]

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return obs["critic"]

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def _update_distribution(self, actor_output: torch.Tensor) -> None:
        if self.state_dependent_std:
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(actor_output, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(actor_output, dim=-2)
                std = torch.exp(log_std)
        else:
            mean = actor_output
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(self.actor(actor_obs))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        output = self.actor(actor_obs)
        if self.state_dependent_std:
            return output[..., 0, :]
        return output

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
