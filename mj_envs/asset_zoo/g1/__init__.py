"""Unitree G1 + decoupled CARTESIAN_HAND configuration for mjlab RL training."""

from .g1_constants import (
    full_collision,
    get_g1_action_scale,
    get_g1_robot_cfg,
    get_spec,
)

__all__ = [
    "get_g1_robot_cfg",
    "get_g1_action_scale",
    "get_spec",
    "full_collision",
]
