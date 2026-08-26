"""PPO adapter wrapping RMAEstimatorCore with velocity estimator supervision.

Replaces the previous ActorCriticHistory subclass with a standalone module that
delegates backbone, normalizer, and vel_head to RMAEstimatorCore. This makes the
estimator stack shareable with FlashSAC (which uses the same core via SequenceActor
with use_velocity_estimator=True).

Architecture:
    actor path:     history(B,S,D) → core.encode() → actor_head → actions
    estimator path: core._last_latent_sg (stop-gradient) → core.vel_head → est_vel(B,3)
    critic path:    privileged_obs → MLP → value

Stop-gradient design (lateral stability):
    vel_head trains on core._last_latent_sg = core._last_latent.detach(). MSE
    gradients flow through vel_head only — the encoder is unaffected by the
    near-zero y-velocity signal (heading_command=True → robot never strafes →
    degenerate y gradient → lateral representation degraded without stop-grad).
    See full rationale in original module docstring history.

Key state_dict layout (NEW — differs from old ActorCriticHistory subclass):
    core.normalizer.*      (was: actor_obs_normalizer.*)
    core.backbone.*        (was: backbone.*)
    core.vel_head.net.*    (was: vel_head.*)
    actor.{weight,bias}    (unchanged)
    critic.*               (unchanged)
    action_std_param       (unchanged)

Use mj_envs/tools/migrate_rma_checkpoint.py to remap old checkpoints.

Export compatibility:
    .backbone → core.backbone (property proxy)
    .vel_head → core.vel_head (property proxy)
    .actor_obs_normalizer → core.normalizer (property proxy)
    These satisfy DeployedPolicyRMAEstimator in export_util.py without changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from flash_sac.models import RMAEstimatorCore


_LOG2PI: float = math.log(2 * math.pi)
_LOG2PIE_05: float = 0.5 * (1.0 + _LOG2PI)


class ActorCriticRMAEstimator(nn.Module):
    """Standalone PPO actor-critic with shared RMAEstimatorCore.

    Mirrors the interface of ActorCriticHistory so CustomPPO / EstimatorPPO
    in custom_runner.py work without modification.

    Args:
        obs: TensorDict providing shapes for actor (B, S, D) and critic (B, D_c).
        obs_groups: Observation group name mapping.
        num_actions: Action space dimension.
        vel_head_hidden_dim: Hidden width of vel_head MLP (default 64).
        encoder_type: "rma_cnn" | "tcn" | "attn_rma_cnn"
        embed_dim: Per-step embedding dim (rma_cnn variants).
        latent_dim: Encoder latent dim.
        actor_hidden_dims: Trunk + actor MLP hidden dims.
        critic_hidden_dims: Critic MLP hidden dims.
        actor_obs_normalization: Own EmpiricalNormalization in core.
        critic_obs_normalization: Normalize critic obs.
        use_layer_norm: Apply LayerNorm in backbone trunk and critic.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        *,
        vel_head_hidden_dim: int = 64,
        vel_head_output_dim: int = 3,
        encoder_type: str = "rma_cnn",
        embed_dim: int = 32,
        latent_dim: int = 128,
        actor_hidden_dims: list[int] = (512, 256, 128),
        critic_hidden_dims: list[int] = (256, 256),
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        actor_obs = obs["actor"]
        if actor_obs.dim() != 3:
            raise ValueError(f"ActorCriticRMAEstimator expects 3D actor obs, got {actor_obs.shape}")
        _, _, frame_dim = actor_obs.shape

        self.core = RMAEstimatorCore(
            frame_dim=frame_dim,
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            backbone_hidden_dims=tuple(actor_hidden_dims),
            activation="elu",
            use_layer_norm=use_layer_norm,
            obs_normalization=actor_obs_normalization,
            vel_head_hidden_dim=vel_head_hidden_dim,
            vel_head_output_dim=vel_head_output_dim,
        )

        self.actor = nn.Linear(self.core.backbone.output_dim, num_actions)

        critic_obs = obs["critic"]
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            from rsl_rl.modules import EmpiricalNormalization
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[critic_obs.shape[-1]])

        critic_layers: list[nn.Module] = []
        in_dim = critic_obs.shape[-1]
        for out_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                critic_layers.append(nn.LayerNorm(out_dim))
            critic_layers.append(nn.ELU())
            in_dim = out_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.action_std_param = nn.Parameter(torch.ones(num_actions))

        # Ephemeral PPO states
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.entropy: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Export compatibility properties (satisfy DeployedPolicyRMAEstimator)
    # ------------------------------------------------------------------

    @property
    def backbone(self):
        return self.core.backbone

    @property
    def vel_head(self):
        return self.core.vel_head

    @property
    def actor_obs_normalizer(self):
        return self.core.normalizer

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        self.core.update_normalization(obs["actor"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs["critic"])

    def _get_actor_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Encode (B, S, D) obs through core; caches latent for EstimatorPPO."""
        return self.core.encode(actor_obs, update_norm=False)

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        actor_obs = obs["actor"] if isinstance(obs, TensorDict) else obs
        return self.actor(self._get_actor_input(actor_obs))

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.action_mean = self.actor(self._get_actor_input(obs["actor"]))
        self.action_std = self.action_std_param.expand_as(self.action_mean)
        actions = self.action_mean + self.action_std * torch.randn_like(self.action_mean)
        self.entropy = (self.action_std.log() + _LOG2PIE_05).sum(dim=-1)
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            -0.5 * ((actions - self.action_mean) / self.action_std).pow(2)
            - self.action_std.log()
            - 0.5 * _LOG2PI
        ).sum(dim=-1)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        critic_obs = obs["critic"]
        if self.critic_obs_normalization:
            critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_estimated_velocity(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Estimate base_lin_vel (B, 3) from proprioceptive history.

        Args:
            obs: TensorDict with 'actor' key, or raw (B, H, D) tensor.
        Returns:
            est_vel: (B, 3) estimated base linear velocity in body frame.
        """
        actor_obs = obs["actor"] if isinstance(obs, TensorDict) else obs
        with torch.no_grad():
            self._get_actor_input(actor_obs)
            return self.core.vel_head(self.core._last_latent)
