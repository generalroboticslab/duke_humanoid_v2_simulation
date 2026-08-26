"""Shared MuJoCo robot viewer utilities.

Provides diagnostic tables for joints/actuators and body masses, plus an
interactive passive viewer with keyboard controls.

Keyboard controls (in the viewer):
  <key>  - Reset to a registered keyframe (key chars defined by caller).
  G      - Toggle gravity on/off.
  I      - Toggle inertia boxes     (red).
  Space  - Pause / resume simulation.
  2 / 3 / 4 / 5
        - Native MuJoCo viewer toggles for geom groups:
            2 = visual meshes, 3 = collision primitives,
            4 = FOV overlay (head-camera cones + spotlights),
            5 = ground plane.

MuJoCo has exactly mjNGROUP=6 geom groups, indices 0..5 — there is no group 6. A geom
declared group="6" is silently CLAMPED to index 5 at render time, and opt.geomgroup[6]
raises IndexError (both verified 2026-08-04). The ground plane therefore lives on group
5, owned by the native digit-5 handler. Group 5 is off in the default geomgroup mask
([1 1 1 0 0 0]), so it is enabled ONCE before the loop rather than per frame — a
per-frame write would clobber the user's digit-5 presses.
"""

import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import mujoco
import mujoco.viewer as viewer
import numpy as np

# Geom group for the standalone viewers' ground plane. 5 is the LAST valid index
# (mjNGROUP=6, so 0..5) and the only one left: 2=visual meshes, 3=collision, 4=FOV
# overlay. Declaring a geom group="6" does NOT get its own toggle — MuJoCo clamps it
# to 5 at render time — so 6 only looked like a free slot.
GROUND_GEOM_GROUP = 5

# RGB colours for the local x/y/z axes drawn by _draw_frame_axes.
_AXIS_COLORS = (
    np.array([1.0, 0.0, 0.0, 1.0]),  # x = red
    np.array([0.0, 1.0, 0.0, 1.0]),  # y = green
    np.array([0.0, 0.0, 1.0, 1.0]),  # z = blue
)

# Maps an axis_frames key to (object type, data position attr, data orientation
# attr). Bodies/sites/geoms expose their world pose under different MjData
# arrays, so each kind is resolved explicitly to avoid cross-type name clashes.
_AXIS_OBJ = {
    "body": (mujoco.mjtObj.mjOBJ_BODY, "xpos", "xmat"),
    # body_com: anchor at the body's centre of mass (xipos) but keep the body
    # frame orientation (xmat). Useful when the joint/frame origin sits far from
    # the visible geometry (e.g. a gimbal link whose origin is at the pivot but
    # whose mass/mesh is out on an arm) so the triad lands on the visible link.
    "body_com": (mujoco.mjtObj.mjOBJ_BODY, "xipos", "xmat"),
    "site": (mujoco.mjtObj.mjOBJ_SITE, "site_xpos", "site_xmat"),
    "geom": (mujoco.mjtObj.mjOBJ_GEOM, "geom_xpos", "geom_xmat"),
}


def _resolve_axis_targets(
    model: mujoco.MjModel, axis_frames: dict[str, Sequence[str]] | None
) -> list[tuple[str, str, int]]:
    """Resolve axis_frames names to (pos_attr, mat_attr, id) triples.

    axis_frames maps "body"/"site"/"geom" -> names. Names absent from the model
    are warned about and skipped (mirrors the foot-geom resolution policy).
    """
    targets: list[tuple[str, str, int]] = []
    if not axis_frames:
        return targets
    for kind, names in axis_frames.items():
        objtype, pos_attr, mat_attr = _AXIS_OBJ[kind]
        for name in names:
            oid = mujoco.mj_name2id(model, objtype, name)
            if oid < 0:
                print(f"  [axis] {kind} '{name}' not found; skipped")
                continue
            targets.append((pos_attr, mat_attr, oid))
    return targets


def _draw_frame_axes(
    scn: mujoco.MjvScene,
    data: mujoco.MjData,
    targets: list[tuple[str, str, int]],
    length: float,
    width: float,
) -> None:
    """Append RGB coordinate-axis arrows for each target to the user scene.

    Resets scn.ngeom, then draws three arrows (x=red, y=green, z=blue) per
    target along its world-frame axis columns. Stops if the scene geom buffer
    fills up.
    """
    scn.ngeom = 0
    for pos_attr, mat_attr, oid in targets:
        pos = getattr(data, pos_attr)[oid]
        mat = getattr(data, mat_attr)[oid].reshape(3, 3)
        for axis in range(3):
            if scn.ngeom >= scn.maxgeom:
                return
            end = pos + mat[:, axis] * length
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                _AXIS_COLORS[axis].astype(np.float32),
            )
            # mjv_initGeom leaves category=0, which the renderer's category mask
            # culls; decor geoms must be tagged mjCAT_DECOR to be drawn.
            g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, pos, end)
            scn.ngeom += 1

# ── Joint/actuator table layout ───────────────────────────────────────────────
# Column widths (chars): joint name, armature, Kp, Kd, action scale, torque
# limit, joint damping, joint friction.
_JNT_COLS = (30, 10, 8, 8, 12, 10, 10, 10)
_JNT_HEADER = (
    f"{'Joint':<{_JNT_COLS[0]}} {'Armature':>{_JNT_COLS[1]}}"
    f" {'Kp':>{_JNT_COLS[2]}} {'Kd':>{_JNT_COLS[3]}}"
    f" {'ActionScale':>{_JNT_COLS[4]}} {'TorqueLim':>{_JNT_COLS[5]}}"
    f" {'JntDamp':>{_JNT_COLS[6]}} {'JntFric':>{_JNT_COLS[7]}}"
)
_JNT_SEP = "─" * len(_JNT_HEADER)

# ── Body mass table layout ────────────────────────────────────────────────────
_MASS_NAME_W, _MASS_VAL_W = 35, 10
_MASS_HEADER = f"{'Link':<{_MASS_NAME_W}} {'Mass (kg)':>{_MASS_VAL_W}}"
_MASS_SEP = "─" * len(_MASS_HEADER)


def print_joint_table(
    model: mujoco.MjModel,
    get_action_scale: Callable[[str], float] | None = None,
) -> None:
    """Print joint/actuator diagnostics in a fixed-width table.

    Columns: joint name, reflected inertia (armature), PD gains (Kp/Kd),
    action scale, torque limit, passive joint damping, and dry friction.
    """
    print(f"\n{_JNT_HEADER}\n{_JNT_SEP}")
    # Build joint→actuator mapping so we iterate in joint order.
    jnt_to_act = {model.actuator_trnid[i, 0]: i for i in range(model.nu)}
    for jnt_id in range(model.njnt):
        if jnt_id not in jnt_to_act:
            continue
        act_id = jnt_to_act[jnt_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        dof_id = model.jnt_dofadr[jnt_id]
        action_scale = get_action_scale(name) if get_action_scale else 1.0
        print(
            f"{name:<{_JNT_COLS[0]}} {model.dof_armature[dof_id]:>{_JNT_COLS[1]}.6f}"
            f" {model.actuator_gainprm[act_id, 0]:>{_JNT_COLS[2]}.2f}"
            f" {-model.actuator_biasprm[act_id, 2]:>{_JNT_COLS[3]}.2f}"
            f" {action_scale:>{_JNT_COLS[4]}.4f}"
            f" {model.actuator_forcerange[act_id, 1]:>{_JNT_COLS[5]}.1f}"
            f" {model.dof_damping[dof_id]:>{_JNT_COLS[6]}.4f}"
            f" {model.dof_frictionloss[dof_id]:>{_JNT_COLS[7]}.4f}"
        )
    print(_JNT_SEP)


def print_mass_table(model: mujoco.MjModel) -> None:
    """Print per-link and total body mass in a fixed-width table."""
    print(f"\n{_MASS_HEADER}\n{_MASS_SEP}")
    total_mass = 0.0
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mass = model.body_mass[body_id]
        total_mass += mass
        if mass > 0:
            print(f"{name:<{_MASS_NAME_W}} {mass:>{_MASS_VAL_W}.4f}")
    print(_MASS_SEP)
    print(f"{'Total Mass':<{_MASS_NAME_W}} {total_mass:>{_MASS_VAL_W}.4f}\n")


def launch_robot_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    # All following arguments must be passed as keywords (no positional).
    *,
    keyframes: dict[str, tuple[str, Callable[[], None]]],
    base_body: str,
    foot_geom_names: list[str],
    get_action_scale: Callable[[str], float] | None = None,
    axis_frames: dict[str, Sequence[str]] | None = None,
    axis_length: float = 0.2,
    axis_width: float = 0.003,
    on_frame: Callable[[], None] | None = None,
) -> None:
    """Launch an interactive MuJoCo viewer with diagnostic stats and keyboard controls.

    Prints joint/actuator and body-mass tables, then opens the viewer window.

    Args:
        model: Compiled MuJoCo model.
        data: Simulation data already initialised to the starting pose.
        keyframes: Maps key character → (label, reset_fn). Each reset_fn
            resets ``data`` in-place with no arguments.
        base_body: Base body name used for the height-above-foot report.
        foot_geom_names: Geom names for finding the lowest foot point.
            Names absent from the model are silently skipped.
        get_action_scale: Optional function (joint name → scale) for the
            joint table ActionScale column. Defaults to 1.0 when None.
        axis_frames: Maps "body"/"site"/"geom" → names whose local coordinate
            axes are drawn (always on) as RGB=xyz arrows. Missing names skipped.
        axis_length: Axis arrow length in metres (long enough to read off pose).
        axis_width: Axis arrow shaft width in metres (thin, non-occluding).
        on_frame: Optional main-thread callback run once per viewer frame before
            simulation. Use for a small companion UI that updates ``data``.
    """
    # ── Diagnostic tables ─────────────────────────────────────────────────────
    print_joint_table(model, get_action_scale)
    print_mass_table(model)

    # ── Resolve IDs ───────────────────────────────────────────────────────────
    foot_geom_ids = [
        gid
        for name in foot_geom_names
        if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
    ]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    axis_targets = _resolve_axis_targets(model, axis_frames)

    # Forward pass so xpos/geom_xpos are populated before reading heights.
    mujoco.mj_forward(model, data)
    base_z = data.xpos[base_id][2]
    foot_z = min(data.geom_xpos[gid][2] for gid in foot_geom_ids)
    print(f"Base ({base_body}) to lowest foot: {base_z - foot_z:.4f} m")

    # ── Keyboard help ─────────────────────────────────────────────────────────
    print("\nKeyboard controls:")
    for key_char, (label, _) in keyframes.items():
        print(f"  {key_char:<6} - Reset to {label}")
    print("  G      - Toggle gravity on/off")
    print("  I      - Toggle inertia boxes     (red)")
    print("  2 / 3  - Native viewer: visual / collision geom groups")
    print("  4 / 5  - Native viewer: FOV overlay / ground plane")
    print("  Space  - Pause / resume simulation\n")

    # ── Viewer state ──────────────────────────────────────────────────────────
    state = SimpleNamespace(
        pending_reset=None,  # key char of the requested reset, or None
        gravity_enabled=False,
        show_inertia=False,
        paused=False,
    )

    def key_callback(key: int) -> None:
        char = chr(key) if 32 <= key < 127 else ""
        if char in keyframes:
            state.pending_reset = char
        elif char == "G":
            state.gravity_enabled = not state.gravity_enabled
            model.opt.gravity[:] = [0, 0, -9.81] if state.gravity_enabled else [0, 0, 0]
            print(f"Gravity {'ENABLED' if state.gravity_enabled else 'DISABLED'}")
        elif char == "I":
            state.show_inertia = not state.show_inertia
            print(f"Inertia boxes: {'ON' if state.show_inertia else 'OFF'}")
        elif key == 32:  # Space
            state.paused = not state.paused
            print(f"Simulation {'PAUSED' if state.paused else 'RESUMED'}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    with viewer.launch_passive(model, data, key_callback=key_callback) as viewer_handle:
        # Ground starts visible. Set ONCE: groups 2/3/4/5 are all owned by the
        # Simulate-native digit-key handlers, so a per-frame write would clobber
        # the user's presses (that bug shipped once already).
        viewer_handle.opt.geomgroup[GROUND_GEOM_GROUP] = 1

        while viewer_handle.is_running():
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = int(state.show_inertia)
            viewer_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0

            if state.pending_reset is not None:
                label, reset_fn = keyframes[state.pending_reset]
                reset_fn()
                print(f"Reset to {label}")
                state.pending_reset = None

            if on_frame is not None:
                on_frame()

            if not state.paused:
                mujoco.mj_step(model, data)

            _draw_frame_axes(
                viewer_handle.user_scn, data, axis_targets, axis_length, axis_width
            )
            viewer_handle.sync()
            # time.sleep(1 / 1000)
