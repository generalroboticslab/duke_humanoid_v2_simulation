# Copyright (c) 2024 General Robotics Lab
# History-aware Actor-Critic architectures for sim-to-real gap bridging.

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization

from flash_sac.models import (
    RMACNNEncoder,
    AttentionPoolRMACNNEncoder,
    TCNEncoder,
    SequenceActorBackbone,
)

# Re-export encoder classes so existing imports from this module still work.
__all__ = [
    "RMACNNEncoder",
    "AttentionPoolRMACNNEncoder",
    "TCNEncoder",
    "SequenceActorBackbone",
    "ActorCriticHistory",
]

# Precomputed constants for analytical Normal distribution (avoids per-call math)
_LOG2PI: float = math.log(2 * math.pi)        # ≈ 1.8379
_LOG2PIE_05: float = 0.5 * (1.0 + _LOG2PI)    # 0.5*(1+log(2π)), for entropy ≈ 1.4189


class ActorCriticHistory(nn.Module):
    """Sequence-aware Actor-Critic network.
    Processes observations as 3D tensors: (Batch, Seq, Dim).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        *,
        encoder_type: str = "rma_cnn",   # "rma_cnn" | "tcn" | "attn_rma_cnn"
        embed_dim: int = 32,             # RMA-style per-step embedding dim
        latent_dim: int = 128,
        actor_hidden_dims: list[int] = (512, 256, 128),
        critic_hidden_dims: list[int] = (256, 256),
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        actor_obs = obs["actor"]
        if actor_obs.dim() != 3:
            raise ValueError(f"ActorCriticHistory expects 3D actor observations, got {actor_obs.shape}")

        _, _, frame_dim = actor_obs.shape
        self.frame_dim = frame_dim

        # Shared backbone: encoder + MLP trunk (no output layer). ELU matches
        # the established activation used across all PPO actor_critic variants.
        self.backbone = SequenceActorBackbone(
            encoder_type=encoder_type,
            input_dim=frame_dim,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dims=actor_hidden_dims,
            activation="elu",
            use_layer_norm=use_layer_norm,
        )
        self.latent_dim = latent_dim

        # Output head: backbone feature → actions (no activation, no squash for PPO).
        self.actor = nn.Linear(self.backbone.output_dim, num_actions)

        # Actor Normalization (applied per-frame before passing to backbone).
        self.actor_obs_normalization = actor_obs_normalization
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[frame_dim])

        # Critic MLP (operates on privileged 2D obs, unchanged).
        critic_obs = obs["critic"]
        self.critic_obs_normalization = critic_obs_normalization
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[critic_obs.shape[-1]])

        critic_layers = []
        in_dim = critic_obs.shape[-1]
        for out_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                critic_layers.append(nn.LayerNorm(out_dim))
            critic_layers.append(nn.ELU())
            in_dim = out_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Fixed std deviation parameter for PPO.
        self.action_std_param = nn.Parameter(torch.ones(num_actions))

        # Ephemeral states required by PPO Runner/Adapters.
        self.action_mean = None
        self.action_std = None
        self.entropy = None

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = obs["actor"]  # (Batch, Seq, Dim)
            self.actor_obs_normalizer.update(actor_obs.reshape(-1, self.frame_dim))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs["critic"])

    def _get_actor_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Normalize per-frame and run through shared backbone."""
        if self.actor_obs_normalization:
            B, S, D = actor_obs.shape
            norm_obs = self.actor_obs_normalizer(actor_obs.reshape(B * S, D)).view(B, S, D)
        else:
            norm_obs = actor_obs
        return self.backbone(norm_obs)  # (B, backbone.output_dim)

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Deterministic action for deployment/evaluation."""
        actor_obs = obs["actor"] if isinstance(obs, TensorDict) else obs
        return self.actor(self._get_actor_input(actor_obs))

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Stochastic action for exploration during training."""
        self.action_mean = self.actor(self._get_actor_input(obs["actor"]))
        self.action_std = self.action_std_param.expand_as(self.action_mean)
        actions = self.action_mean + self.action_std * torch.randn_like(self.action_mean)
        self.entropy = (self.action_std.log() + _LOG2PIE_05).sum(dim=-1)
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of actions under the current distribution."""
        return (
            -0.5 * ((actions - self.action_mean) / self.action_std).pow(2)
            - self.action_std.log()
            - 0.5 * _LOG2PI
        ).sum(dim=-1)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate state value using critic."""
        critic_obs = obs["critic"]
        if self.critic_obs_normalization:
            critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)
