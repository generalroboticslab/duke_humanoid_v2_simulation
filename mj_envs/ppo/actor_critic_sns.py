# Copyright (c) 2024 General Robotics Lab
# Smooth Neural Surrogate (SNS) Actor-Critic for Lipschitz-constrained policies.
#
# Reference: "Smooth Neural Surrogates" - constrains network Lipschitz constant
# for robust sim-to-real transfer.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # FIXED: Added missing import
from typing import Any
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class SNSLinear(nn.Module):
    """Linear layer with learnable Lipschitz constant (SNS parameterization).

    Each layer learns a scalar θ_c that determines its Lipschitz budget.
    Weights are row-normalized during forward pass to respect this budget.

    The ∞-norm (max row sum) is used as the Lipschitz constant estimate:
        ||W||_∞ = max_i Σ_j |W_ij|

    During forward: W_normalized = W * min(1, c / ||W||_∞) row-wise
    where c = exp(θ_c) is the learned Lipschitz constant.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Learnable Lipschitz constant: c = exp(theta_c)
        # Initialized to match current weight norm
        self.theta_c = nn.Parameter(torch.tensor(0.0))

        # Inference mode flag - when True, weights are pre-normalized
        self._inference_mode = False

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming and set θ_c to match."""
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # Initialize θ_c so that c = ||W||_∞ (no initial scaling)
        with torch.no_grad():
            row_norms = self.weight.abs().sum(dim=1)
            self.theta_c.fill_(torch.log(row_norms.max()).item())

    @property
    def c(self) -> torch.Tensor:
        """Current Lipschitz constant (always positive via exp)."""
        return torch.exp(self.theta_c)

    def get_row_norms(self) -> torch.Tensor:
        """Compute ∞-norm per row: ||W_i||_1 = Σ_j |W_ij|"""
        return self.weight.abs().sum(dim=1)

    def get_normalized_weight(self) -> torch.Tensor:
        """Return weight matrix normalized to respect Lipschitz bound c."""
        if self._inference_mode:
            return self.weight  # Already normalized

        # BYPASS: Skip normalization during training to test if it causes slowdown
        # Set SNSLinear.BYPASS_NORMALIZATION = True to disable
        if getattr(SNSLinear, 'BYPASS_NORMALIZATION', False):
            return self.weight

        row_norms = self.get_row_norms()
        # Scale down rows that exceed budget (never scale up)
        scale = torch.clamp(self.c / (row_norms + 1e-8), max=1.0)
        return self.weight * scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_mode or getattr(SNSLinear, 'BYPASS_NORMALIZATION', False):
            return F.linear(x, self.weight, self.bias)
        # Scale output instead of weight matrix: diag(s)·W·x = s ⊙ (W·x)
        row_norms = self.get_row_norms()
        scale = torch.clamp(self.c / (row_norms + 1e-8), max=1.0)
        out = F.linear(x, self.weight, None)
        return out * scale + self.bias if self.bias is not None else out * scale

    def set_inference_mode(self, mode: bool = True):
        """Bake normalization into weights for zero-overhead inference."""
        if mode and not self._inference_mode:
            with torch.no_grad():
                self.weight.copy_(self.get_normalized_weight())
        self._inference_mode = mode

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, c={self.c.item():.3f}'


class SNSMLP(nn.Module):
    """MLP with SNS layers for Lipschitz-constrained output.

    The total Lipschitz constant is bounded by C = ∏_i c_i where c_i is
    each layer's learned constant. Training adds a soft penalty when C > C_target.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function (use smooth ones: mish, tanh, elu)
        lipschitz_ub: Target upper bound C_target. If None, computed from initial C.
        lipschitz_scale: When lipschitz_ub is None, target = C_init * lipschitz_scale.
                        Default 0.01 means target is 100x smaller than initial.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        lipschitz_ub: float | None = None,
        lipschitz_scale: float = 0.01,
    ):
        super().__init__()

        # Build layers first (need them to compute initial C)
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(SNSLinear(dims[i], dims[i + 1]))

        self.activation = resolve_nn_activation(activation)
        self.n_layers = len(self.layers)

        # Compute initial Lipschitz constant (product of all layer constants)
        with torch.no_grad():
            C_init = 1.0
            per_layer_c = []
            for layer in self.layers:
                c_i = layer.c.item()
                C_init *= c_i
                per_layer_c.append(c_i)

        # Set lipschitz_ub relative to initial C if not provided
        if lipschitz_ub is None:
            self.lipschitz_ub = C_init * lipschitz_scale
            
            # FIXED: Distribute the scale headroom to the layers immediately!
            # Otherwise, c starts equal to |W|, creating a tight bottleneck that
            # stifles initial learning because d(theta)/dL is 0 when inactive.
            if lipschitz_scale > 1.0:
                scale_per_layer = lipschitz_scale ** (1.0 / self.n_layers)
                log_scale = math.log(scale_per_layer)
                for layer in self.layers:
                    # Direct in-place modification of the parameter
                    with torch.no_grad():
                        layer.theta_c.add_(log_scale)
                print(f"[SNSMLP] Distributing headroom: Scaled each layer c by {scale_per_layer:.2f} (Log scale +{log_scale:.4f})")

            print(f"[SNSMLP] C_init={C_init:.2f} (per-layer: {[f'{c:.2f}' for c in per_layer_c]})")
            print(f"[SNSMLP] lipschitz_ub=C_init*{lipschitz_scale}={self.lipschitz_ub:.2f}")
        else:
            self.lipschitz_ub = lipschitz_ub
            print(f"[SNSMLP] C_init={C_init:.2f}, lipschitz_ub={lipschitz_ub:.2f} (explicit)")
            if lipschitz_ub > C_init:
                print(f"[SNSMLP] WARNING: lipschitz_ub > C_init, constraint may never activate!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:  # No activation after last layer
                x = self.activation(x)
        return x

    def get_lipschitz_constant(self) -> torch.Tensor:
        """Compute current total Lipschitz constant C = ∏_i c_i."""
        C = torch.ones(1, device=self.layers[0].theta_c.device)
        for layer in self.layers:
            C = C * layer.c
        return C

    def get_lipschitz_residual(self) -> torch.Tensor:
        """Compute soft constraint residual: max(1, C / C_target).

        Returns 1.0 when C ≤ C_target (constraint satisfied).
        Returns C/C_target > 1 when violated (penalty grows linearly).
        """
        C = self.get_lipschitz_constant()
        return torch.clamp(C / self.lipschitz_ub, min=1.0)

    def set_inference_mode(self, mode: bool = True):
        """Bake normalization into all layers for deployment."""
        for layer in self.layers:
            layer.set_inference_mode(mode)


class ActorCriticSNS(nn.Module):
    """Actor-Critic with Lipschitz-constrained actor (SNS).

    The actor uses SNSMLP for smooth, bounded sensitivity to observations.
    The critic uses standard MLP (no constraint needed - not deployed).

    Args:
        lipschitz_ub: Target Lipschitz bound for actor. If None, computed from
                     initial network Lipschitz constant using lipschitz_scale.
        lipschitz_scale: When lipschitz_ub is None, target = C_init * scale.
                        Default 0.01 means target is 100x smaller than initial.
        All other args passed to base ActorCritic.
    """

    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        lipschitz_ub: float | None = None,
        lipschitz_scale: float = 0.01,
        **kwargs: dict[str, Any],
    ) -> None:
        # Initialize nn.Module directly (skip ActorCritic.__init__ which uses different MLP)
        nn.Module.__init__(self)
        self.is_recurrent = False

        if kwargs:
            print(f"[ActorCriticSNS] Ignoring unknown args: {list(kwargs.keys())}")

        num_actor_obs = obs["actor"].shape[-1]
        num_critic_obs = obs["critic"].shape[-1]

        # === ACTOR: SNS-constrained MLP ===
        self.actor = SNSMLP(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dims=tuple(actor_hidden_dims),
            activation=activation,
            lipschitz_ub=lipschitz_ub,
            lipschitz_scale=lipschitz_scale,
        )
        # Initialize last layer to near-zero for stable start (matching LayerNorm baseline behavior)
        # This prevents large random actions at the start of training
        with torch.no_grad():
            last_layer = self.actor.layers[-1]
            last_layer.weight.mul_(0.01)
            if last_layer.bias is not None:
                last_layer.bias.zero_()
            # Also reset theta_c to match the small weight norm
            row_norms = last_layer.weight.abs().sum(dim=1)
            last_layer.theta_c.fill_(torch.log(row_norms.max() + 1e-8).item())
        print(f"[ActorCriticSNS] Actor: SNSMLP(lipschitz_ub={self.actor.lipschitz_ub:.2f})")
        print(f"[ActorCriticSNS] Initialized last layer weights with scale 0.01")

        # === CRITIC: SNS-constrained MLP ===
        self.critic = SNSMLP(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dims=tuple(critic_hidden_dims),
            activation=activation,
            lipschitz_ub=lipschitz_ub,
            lipschitz_scale=lipschitz_scale,
        )
        print(f"[ActorCriticSNS] Critic: SNSMLP(lipschitz_ub={self.critic.lipschitz_ub:.2f})")

        # Observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        # Action noise (scalar std)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None

    def get_lipschitz_residual(self) -> torch.Tensor:
        """Get combined Lipschitz residual from actor and critic."""
        return self.actor.get_lipschitz_residual() + self.critic.get_lipschitz_residual()

    def get_lipschitz_constant(self) -> torch.Tensor:
        """Get current Lipschitz constant from actor (primary, used for logging)."""
        return self.actor.get_lipschitz_constant()

    def get_critic_lipschitz_constant(self) -> torch.Tensor:
        """Get current Lipschitz constant from critic."""
        return self.critic.get_lipschitz_constant()

    def set_inference_mode(self, mode: bool = True):
        """Bake weight normalization into actor and critic for deployment."""
        self.actor.set_inference_mode(mode)
        self.critic.set_inference_mode(mode)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return obs["actor"]

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return obs["critic"]

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_obs_normalizer(self.get_actor_obs(obs))

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.critic_obs_normalizer(self.get_critic_obs(obs))

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
        """Update observation normalizers."""
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Sample action from policy distribution."""
        actor_obs = self._get_actor_obs(obs)
        mean = self.actor(actor_obs)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = torch.distributions.Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action for inference."""
        actor_obs = self._get_actor_obs(obs)
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Evaluate value function."""
        critic_obs = self._get_critic_obs(obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)
