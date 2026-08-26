import math

import numpy as np
import torch

_ALPHA  = math.pi / 12   # 15° hip joint tilt
_D_LINK = 195.0          # mm — thigh = shank length (symmetric scissor mechanism)

_COS_A  = math.cos(_ALPHA)
_SIN_A  = math.sin(_ALPHA)
_INV_2D = 1.0 / (2.0 * _D_LINK)  # reciprocal for z_travel scaling: multiply > divide

LEG_JOINT_NAMES = (
    'left_hip_1_joint',  'left_hip_2_joint',  'left_hip_3_joint',
    'left_knee_joint',   'left_ankle_1_joint', 'left_ankle_2_joint',
    'right_hip_1_joint', 'right_hip_2_joint',  'right_hip_3_joint',
    'right_knee_joint',  'right_ankle_1_joint','right_ankle_2_joint',
)


def vertical_translate_lower_body_joints(z_travel, is_forward_bend=True):
    """Leg joint angles for a symmetric vertical squat.

    Args:
        z_travel (float): body drop in mm, range [-150, 0].
        is_forward_bend (bool): True = human-like (knee forward),
                                False = ostrich-like (knee backward).

    Geometry: equal-length thigh/shank scissor — vertical drop = 2*D_LINK*(1-cos(phi)):
      phi = arccos(1 + z_travel / (2 * D_LINK))

    Hip IK: the 15° joint tilt (alpha) couples all three joints; pure t1=phi is wrong.
    Exact closed-form solution for target = thigh rotates forward by phi from rest:
      t1  = atan2(-sin(phi),  cos(alpha)*cos(phi))
      t2  = alpha - asin(sin(alpha)*cos(phi))       # tilt correction; depends on cos(phi) only
      t3  = atan2(-cos(alpha), sin(alpha)*sin(phi)) + pi/2

    Forward/backward bend flips t1, t3, knee, ankle — t2 is sign-invariant because
    it depends only on cos(phi), which is the same for ±phi.

    Right leg = negated left leg (mirrored joint axes).

    Returns: dict of joint names to angles (radians), ordered as LEG_JOINT_NAMES.
    """
    if not (-150 <= z_travel <= 0):
        raise ValueError(f"z_travel must be in [-150, 0], got {z_travel}")

    # --- geometry ---------------------------------------------------------
    phi = math.acos(max(-1.0, min(1.0, 1.0 + z_travel / (2.0 * _D_LINK))))

    # --- hip IK -----------------------------------------------------------
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    t1 = math.atan2(-sin_phi, _COS_A * cos_phi)
    t2 = _ALPHA - math.asin(max(-1.0, min(1.0, _SIN_A * cos_phi)))
    t3 = math.atan2(-_COS_A, _SIN_A * sin_phi) + math.pi / 2

    # --- assemble ---------------------------------------------------------
    s = 1 if is_forward_bend else -1
    left_leg = np.array([
        -s * t1,
         t2,      # sign-invariant
        -s * t3,
         s * 2 * phi,
         s * phi,
         0.0,
    ], dtype=np.float32)
    right_leg = -left_leg

    return {name: val for name, val in zip(LEG_JOINT_NAMES, np.concatenate([left_leg, right_leg]))}


def vertical_translate_lower_body_joints_batch(
    z_travel: torch.Tensor,
    is_forward_bend: bool = True,
) -> torch.Tensor:
    """Vectorized torch version of vertical_translate_lower_body_joints.

    Args:
        z_travel: [B] float tensor, body drop in mm, values in [-150, 0].
        is_forward_bend: True = human-like (knee forward), False = ostrich-like.

    Returns:
        [B, 12] float tensor, joint angles in radians, ordered as LEG_JOINT_NAMES.
    """
    cos_phi = (1.0 + z_travel * _INV_2D).clamp(-1.0, 1.0)         # [B]
    phi     = torch.acos(cos_phi)                                   # [B]
    sin_phi = torch.sin(phi)                                        # [B]

    t1 = torch.atan2(-sin_phi, _COS_A * cos_phi)                   # [B]
    t2 = _ALPHA - torch.asin((_SIN_A * cos_phi).clamp(-1.0, 1.0)) # [B]
    # identity: atan2(y,x) + π/2 = atan2(x,-y), eliminates [B] full_like alloc + scalar add
    t3 = torch.atan2(_SIN_A * sin_phi, cos_phi.new_full((), _COS_A))  # [B]

    s = 1.0 if is_forward_bend else -1.0
    out = torch.empty(z_travel.shape[0], 12, dtype=z_travel.dtype, device=z_travel.device)
    # left leg (cols 0-5), right leg = -left leg (cols 6-11), written inline
    out[:, 0]  = -s * t1;         out[:, 6]  = s * t1
    out[:, 1]  = t2;              out[:, 7]  = -t2
    out[:, 2]  = -s * t3;         out[:, 8]  = s * t3
    out[:, 3]  = s * 2.0 * phi;  out[:, 9]  = -s * 2.0 * phi
    out[:, 4]  = s * phi;         out[:, 10] = -s * phi
    out[:, 5]  = 0.0;             out[:, 11] = 0.0
    return out


if __name__ == "__main__":
    for mode, flag in [("forward (human)", True), ("backward (ostrich)", False)]:
        print(f"\n--- {mode} ---")
        j = vertical_translate_lower_body_joints(-60, is_forward_bend=flag)
        print(f"{'Joint':<25} {'Angle (rad)':>12} {'Angle (deg)':>12}")
        print("-" * 51)
        for name, angle in j.items():
            print(f"{name:<25} {angle:12.4f} {math.degrees(angle):12.2f}")
