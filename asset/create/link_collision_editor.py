#!/usr/bin/env python3
"""Generic per-link collision-geom editor for any MuJoCo MJCF.

Interactive GUI that lets you add, remove, and pose collision-geometry primitives
(sphere, cylinder, capsule, box) on any body in the model. A dearpygui panel runs
alongside a MuJoCo passive viewer — edits update live.

Works with any robot MJCF that follows the repo group convention:
    group 1 or 2 = visual meshes, group 3 = collision primitives, group 5 = rack proxy.
These defaults can be overridden with --visual-groups and --collision-group.

Supported primitive types:
    sphere   — radius only
    cylinder — radius + halflen, orientation
    capsule  — radius + halflen, orientation (rendered as capsule)
    box      — 3 half-extents, orientation

Output:
    Nothing is auto-written to the source XML. Press "Print ALL" to dump the current
    collision state (XML snippet per body) to console AND clipboard. Optionally pass
    --write-xml to overwrite the source <geom> tags in-place.

Usage:
    python link_collision_editor.py --xml path/to/robot.xml
    python link_collision_editor.py --xml path/to/robot.xml --visual-groups 1
    python link_collision_editor.py --xml path/to/robot.xml --write-xml

Requirements:
    pip install dearpygui trimesh mujoco

Previous scratch tool: /tmp/t1_link_editor.py (Booster T1 handoff, not in git).
"""
from __future__ import annotations

import argparse
import copy
import os
import pathlib
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np
from mujoco import viewer

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False

try:
    import dearpygui.dearpygui as dpg
    _HAS_DPG = True
except ImportError:
    _HAS_DPG = False

# ──────────────────────────────────────────────────────────────────────────────
# Standard group roles (matches asset/create/export_mjspec_to_urdf.py)
# ──────────────────────────────────────────────────────────────────────────────
# group 1 or 2 = visual mesh, group 3 = collision primitives,
# group 4 = FOV / camera overlay, group 5 = rack proxy (supersedes g3 per body).
_DEFAULT_VISUAL_GROUPS = {1, 2}
_DEFAULT_COLLISION_GROUP = 3
_DEFAULT_DISABLED_GROUP = 5  # Hidden geoms placed here; reuses group 5 (viewer toggle)

# Geom type names for display and XML output.
_GEOM_TYPE_NAME = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
}
_GEOM_NAME_TYPE = {v: k for k, v in _GEOM_TYPE_NAME.items()}

# Max placeholder slots per body (2 sphere + 2 cylinder + 2 capsule + 2 box = 8).
_SLOTS_PER_TYPE = 2
_SLOT_TYPES = ("sphere", "cylinder", "capsule", "box")
_MAX_SLOTS = _SLOTS_PER_TYPE * len(_SLOT_TYPES)

# Axes palette for body-frame highlight arrows.
_AXIS_COLORS = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → intrinsic XYZ Euler angles (rad)."""
    w, x, y, z = q
    ex = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    ey = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    ez = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([ex, ey, ez])


def _euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles (rad) → quaternion (w,x,y,z)."""
    ex, ey, ez = euler
    cx, sx = np.cos(ex / 2), np.sin(ex / 2)
    cy, sy = np.cos(ey / 2), np.sin(ey / 2)
    cz, sz = np.cos(ez / 2), np.sin(ez / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_from_z_to(u: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating +Z onto unit vector u."""
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    c = np.dot(z, u)
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 1, 0, 0])
    ax = np.cross(z, u)
    ax /= np.linalg.norm(ax)
    ang = np.arccos(np.clip(c, -1, 1))
    return np.array([np.cos(ang / 2), *(ax * np.sin(ang / 2))])


_MIRROR_TOKEN_PAIRS = {
    "left": "right", "right": "left",
    "Left": "Right", "Right": "Left",
    "LEFT": "RIGHT", "RIGHT": "LEFT",
    "l": "r", "r": "l",
    "L": "R", "R": "L",
}


def _mirror_body_name(name: str) -> Optional[str]:
    """Best-effort left/right counterpart of *name* via underscore-token swap.

    Handles the naming families seen across robots in this repo: prefix
    (``l_hip_aa_link``), infix (``arm_left_1_link``), and single-letter
    token (``ankle_L_Link``). Returns None if no side token is found —
    caller should fall back to manual selection.
    """
    tokens = name.split("_")
    for i, tok in enumerate(tokens):
        if tok in _MIRROR_TOKEN_PAIRS:
            mirrored = tokens.copy()
            mirrored[i] = _MIRROR_TOKEN_PAIRS[tok]
            return "_".join(mirrored)
    return None


def _mirror_pos_quat(pos: np.ndarray, quat: np.ndarray):
    """Reflect a body-local (pos, quat) across the body's local XZ-plane.

    Assumes sibling left/right bodies share the same local-axis convention
    and differ only by this y-flip (true for every left/right pair checked
    in this repo: l_/r_, left_/right_, _L_/_R_ families) — not a general
    guarantee for arbitrary MJCF. y-flip of a proper rotation quaternion
    negates the two imaginary components off the mirror-normal axis (x, z);
    w and the y-component stay put.
    """
    mp = pos.copy()
    mp[1] = -mp[1]
    mq = quat.copy()
    mq[1] = -mq[1]
    mq[3] = -mq[3]
    return mp, mq


def _fit_from_mesh(mesh_path: str, axis_local: np.ndarray,
                   r_pct: float = 90, r_margin: float = 0.003,
                   l_pct: float = 100, l_margin: float = 0.003):
    """Fit a capsule/cylinder to a mesh along *axis_local*.

    Returns (radius, halflen, pos_local, quat_local).
    """
    if not _HAS_TRIMESH:
        raise RuntimeError("trimesh not installed — auto-fit unavailable")
    v = trimesh.load(mesh_path).vertices
    u = axis_local / np.linalg.norm(axis_local)
    t = v @ u
    perp = v - np.outer(t, u)
    r = np.linalg.norm(perp, axis=1)
    radius = float(np.percentile(r, r_pct) + r_margin)
    tmin, tmax = np.percentile(t, 100 - l_pct), np.percentile(t, l_pct)
    halflen = float((tmax - tmin) / 2 + l_margin)
    pos = ((tmax + tmin) / 2) * u
    return radius, halflen, pos, _quat_from_z_to(u)


# ──────────────────────────────────────────────────────────────────────────────
# Geom slot state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SlotState:
    """Mutable state for one collision-geom slot."""
    bid: int
    gid: int
    geom_type: str  # "sphere", "cylinder", "capsule", "box"
    pos: np.ndarray
    quat: np.ndarray
    size: np.ndarray  # [radius] or [radius, halflen] or [hx, hy, hz]
    enabled: bool
    euler_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def clone(self) -> SlotState:
        return SlotState(
            bid=self.bid, gid=self.gid, geom_type=self.geom_type,
            pos=self.pos.copy(), quat=self.quat.copy(), size=self.size.copy(),
            enabled=self.enabled, euler_deg=self.euler_deg.copy(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Editor core
# ──────────────────────────────────────────────────────────────────────────────

class CollisionEditor:
    """Robot-generic collision-geom editor with dearpygui + MuJoCo viewer."""

    def __init__(self, xml_path: str, assets_dir: Optional[str] = None,
                 visual_groups: Optional[set[int]] = None,
                 collision_group: int = _DEFAULT_COLLISION_GROUP,
                 disabled_group: int = _DEFAULT_DISABLED_GROUP,
                 write_xml: bool = False):
        self.xml_path = os.path.abspath(xml_path)
        self.xml_dir = os.path.dirname(self.xml_path)
        self.xml_basename = os.path.basename(self.xml_path)
        self.write_xml = write_xml

        self.visual_groups = visual_groups or _DEFAULT_VISUAL_GROUPS
        self.collision_group = collision_group
        self.disabled_group = disabled_group

        # Resolve assets directory.
        if assets_dir:
            self.assets_dir = os.path.abspath(assets_dir)
        else:
            for candidate in ("meshes", "assets", "mesh"):
                p = os.path.join(self.xml_dir, candidate)
                if os.path.isdir(p):
                    self.assets_dir = p
                    break
            else:
                self.assets_dir = self.xml_dir

        # Undo stack.
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        # Build augmented model.
        self._build_augmented_model()
        self._discover_bodies()

        # Threading.
        self.lock = threading.Lock()
        self.selected_body = self.body_names[0] if self.body_names else ""
        self.selected_slot_idx = 0
        self._last_perturb_select = -1
        self.cur_state: Optional[SlotState] = None
        self.cur_jid: Optional[int] = None
        if self.body_names:
            bid = self._bid(self.selected_body)
            slots = self._slot_gids(bid)
            if slots:
                self.cur_state = self._load_state(bid, slots[0])
            self.cur_jid = self._body_jid(bid)

    # ── Model augmentation ──────────────────────────────────────────────────

    def _build_augmented_model(self):
        """Parse source XML, inject placeholder collision geoms, compile.

        Uses a two-pass approach:
          1. Compile the original XML to discover which bodies have visual-group
             geoms (this correctly resolves MuJoCo class inheritance, e.g.
             Apollo's ``visual_dark`` inheriting ``group=1`` from ``visual``).
          2. Inject placeholder collision-geom slots into matching XML body
             elements, then recompile the augmented XML.
        """
        # --- Pass 1: compile source to discover visual bodies ----------------
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        # Temporarily point meshdir at assets for the probe compile.
        compiler = root.find("compiler")
        orig_meshdir = None
        if compiler is not None:
            orig_meshdir = compiler.get("meshdir")
            compiler.set("meshdir", self.assets_dir)
        else:
            compiler = ET.SubElement(root, "compiler")
            compiler.set("meshdir", self.assets_dir)
        probe_file = tempfile.NamedTemporaryFile(
            suffix=".xml", prefix="collision_editor_probe_", delete=False, mode="w"
        )
        tree.write(probe_file.name)
        probe_file.close()
        try:
            probe_model = mujoco.MjModel.from_xml_path(probe_file.name)
        finally:
            os.unlink(probe_file.name)

        # Collect body names that have at least one visual-group geom.
        visual_body_names: set[str] = set()
        for gid in range(probe_model.ngeom):
            if probe_model.geom_group[gid] in self.visual_groups:
                bid = probe_model.geom_bodyid[gid]
                name = probe_model.body(bid).name
                if name:
                    visual_body_names.add(name)
        del probe_model

        # Not every robot MJCF in this repo defines a `collision` default
        # class (apollo/talos/g1 do; gr3 uses bare group="3" geoms instead).
        # Emitting class="collision" against a file without that default
        # produces an unloadable XML ("unknown default class name").
        self.has_collision_class = any(
            d.get("class") == "collision" for d in root.iter("default")
        )

        # --- Pass 2: inject placeholders into those bodies -------------------
        # Re-parse (clean tree without probe meshdir mutation).
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        def _process(body_el: ET.Element):
            body_name = body_el.get("name", "")
            if body_name in visual_body_names:
                # Add placeholder slots: 2 of each type.
                for type_name in _SLOT_TYPES:
                    for _ in range(_SLOTS_PER_TYPE):
                        g = ET.SubElement(body_el, "geom")
                        g.set("type", type_name)
                        g.set("group", str(self.disabled_group))
                        g.set("contype", "0")
                        g.set("conaffinity", "0")
                        if type_name == "sphere":
                            g.set("size", "0.02")
                        elif type_name in ("cylinder", "capsule"):
                            g.set("size", "0.02 0.02")
                        elif type_name == "box":
                            g.set("size", "0.02 0.02 0.02")
                        # Use zero pos — the runtime sameframe=0 override
                        # (applied after compilation) ensures edits update live.
                        g.set("pos", "0 0 0")
                        g.set("rgba", "0.9 0.3 0.2 0.7")
            for child in body_el.findall("body"):
                _process(child)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for top in worldbody.findall("body"):
                _process(top)

        # Point meshdir at assets.
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", self.assets_dir)
        else:
            compiler = ET.SubElement(root, "compiler")
            compiler.set("meshdir", self.assets_dir)

        # Write augmented XML to tempfile, compile.
        self._aug_xml = tempfile.NamedTemporaryFile(
            suffix=".xml", prefix="collision_editor_", delete=False, mode="w"
        )
        tree.write(self._aug_xml.name)
        self._aug_xml.close()

        self.model = mujoco.MjModel.from_xml_path(self._aug_xml.name)
        self.data = mujoco.MjData(self.model)
        # Reset to first keyframe if available, else default.
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Disable sameframe optimization for ALL editable geoms.
        for gid in range(self.model.ngeom):
            grp = self.model.geom_group[gid]
            if grp == self.collision_group or grp == self.disabled_group:
                self.model.geom_sameframe[gid] = 0

    # ── Body / slot discovery ───────────────────────────────────────────────

    def _discover_bodies(self):
        """Find all bodies with visual geoms → these get collision editing."""
        m = self.model
        seen = set()
        self.body_names: list[str] = []
        for bid in range(m.nbody):
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == bid and m.geom_group[gid] in self.visual_groups:
                    name = m.body(bid).name
                    if name and name not in seen:
                        self.body_names.append(name)
                        seen.add(name)
                    break

    def _bid(self, body_name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def _body_jid(self, bid: int) -> Optional[int]:
        """First scalar (hinge/slide) joint id on this body, else None.

        Excludes free/ball joints — those aren't representable as a single
        angle slider (all bodies here have 0 or 1 joint; base_link's
        floating_base freejoint is the one exclusion in this repo)."""
        m = self.model
        if m.body_jntnum[bid] == 0:
            return None
        jid = int(m.body_jntadr[bid])
        if m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            return jid
        return None

    def _slot_gids(self, bid: int) -> list[int]:
        """Editable collision-geom slots for a body (collision or disabled group, not box-only)."""
        m = self.model
        return sorted(
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] == bid
            and m.geom_group[g] in (self.collision_group, self.disabled_group)
        )

    def _load_state(self, bid: int, gid: int) -> SlotState:
        m = self.model
        type_int = m.geom_type[gid]
        type_name = _GEOM_TYPE_NAME.get(type_int, "sphere")
        pos = m.geom_pos[gid].copy()
        quat = m.geom_quat[gid].copy()
        size = m.geom_size[gid].copy()
        enabled = bool(m.geom_group[gid] == self.collision_group)
        euler_deg = np.rad2deg(_quat_to_euler_xyz(quat))
        return SlotState(bid=bid, gid=gid, geom_type=type_name,
                         pos=pos, quat=quat, size=size, enabled=enabled,
                         euler_deg=euler_deg)

    # ── Mesh / axis helpers ─────────────────────────────────────────────────

    def _mesh_path_for_body(self, bid: int) -> Optional[str]:
        m = self.model
        for gid in range(m.ngeom):
            if (m.geom_bodyid[gid] == bid
                    and m.geom_group[gid] in self.visual_groups
                    and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = m.geom_dataid[gid]
                if mesh_id >= 0:
                    mesh_name = m.mesh(mesh_id).name
                    for ext in (".stl", ".STL", ".obj", ".OBJ", ".ply", ".PLY"):
                        path = os.path.join(self.assets_dir, f"{mesh_name}{ext}")
                        if os.path.exists(path):
                            return path
                        path = os.path.join(self.assets_dir, mesh_name)
                        if os.path.exists(path):
                            return path
        return None

    def _joint_axis_local(self, bid: int) -> Optional[np.ndarray]:
        m = self.model
        jadr, jnum = m.body_jntadr[bid], m.body_jntnum[bid]
        return m.jnt_axis[jadr].copy() if jnum > 0 else None

    def _bone_axis_local(self, bid: int) -> Optional[np.ndarray]:
        m = self.model
        for cb in range(m.nbody):
            if m.body_parentid[cb] == bid:
                child_pos = m.body_pos[cb]
                if np.linalg.norm(child_pos) > 1e-6:
                    return child_pos.copy()
        return None

    # ── Geom XML generation ─────────────────────────────────────────────────

    def _geom_xml_line(self, gid: int) -> Optional[str]:
        """Generate MJCF <geom> line for one geom slot."""
        m = self.model
        if m.geom_group[gid] == self.disabled_group:
            return None
        t = m.geom_type[gid]
        p = m.geom_pos[gid]
        q = m.geom_quat[gid]
        s = m.geom_size[gid]
        type_name = _GEOM_TYPE_NAME.get(t, "sphere")

        def _pos_str():
            return f'pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"'

        def _quat_str():
            if np.allclose(q, [1, 0, 0, 0], atol=1e-4):
                return ""
            return f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '

        # class="collision" only if the source defines that default class
        # (apollo/talos/g1); otherwise fall back to an explicit group attr
        # (gr3-style bare geoms) so the written XML stays loadable.
        class_or_group = ('class="collision"' if self.has_collision_class
                          else f'group="{self.collision_group}"')

        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return f'<geom {class_or_group} size="{s[0]:.4f}" {_pos_str()} type="sphere"/>'
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return f'<geom {class_or_group} size="{s[0]:.4f} {s[1]:.4f}" {_pos_str()} {_quat_str()}type="cylinder"/>'
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return f'<geom {class_or_group} size="{s[0]:.4f} {s[1]:.4f}" {_pos_str()} {_quat_str()}type="capsule"/>'
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            return f'<geom {class_or_group} size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" {_pos_str()} {_quat_str()}type="box"/>'
        return None

    def print_all(self) -> str:
        """Dump current collision state (all enabled slots) per body. Returns text."""
        lines = [f"--- collision state for {self.xml_basename} ---"]
        for bn in self.body_names:
            bid = self._bid(bn)
            geom_lines = []
            for gid in self._slot_gids(bid):
                xl = self._geom_xml_line(gid)
                if xl is not None:
                    geom_lines.append(xl)
            if not geom_lines:
                lines.append(f"{bn:30s} (no collision geom)")
            else:
                lines.append(f"{bn:30s} {geom_lines[0]}")
                for gl in geom_lines[1:]:
                    lines.append(f"{'':30s} {gl}")
        text = "\n".join(lines)
        print("\n" + text)
        return text

    # ── Undo/redo ───────────────────────────────────────────────────────────

    def _push_undo(self):
        if self.cur_state is not None:
            self._undo_stack.append({
                "gid": self.cur_state.gid,
                "pos": self.model.geom_pos[self.cur_state.gid].copy(),
                "quat": self.model.geom_quat[self.cur_state.gid].copy(),
                "size": self.model.geom_size[self.cur_state.gid].copy(),
                "group": int(self.model.geom_group[self.cur_state.gid]),
            })
            self._redo_stack.clear()

    def _push_undo_gid(self, gid: int):
        """Push undo snapshot for an arbitrary gid (not just cur_state's)."""
        m = self.model
        self._undo_stack.append({
            "gid": gid,
            "pos": m.geom_pos[gid].copy(),
            "quat": m.geom_quat[gid].copy(),
            "size": m.geom_size[gid].copy(),
            "group": int(m.geom_group[gid]),
        })
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        gid = snap["gid"]
        # Push current to redo.
        self._redo_stack.append({
            "gid": gid,
            "pos": self.model.geom_pos[gid].copy(),
            "quat": self.model.geom_quat[gid].copy(),
            "size": self.model.geom_size[gid].copy(),
            "group": int(self.model.geom_group[gid]),
        })
        self.model.geom_pos[gid] = snap["pos"]
        self.model.geom_quat[gid] = snap["quat"]
        self.model.geom_size[gid] = snap["size"]
        self.model.geom_group[gid] = snap["group"]
        if self.cur_state and self.cur_state.gid == gid:
            self.cur_state = self._load_state(self.cur_state.bid, gid)

    def _redo(self):
        if not self._redo_stack:
            return
        snap = self._redo_stack.pop()
        gid = snap["gid"]
        self._undo_stack.append({
            "gid": gid,
            "pos": self.model.geom_pos[gid].copy(),
            "quat": self.model.geom_quat[gid].copy(),
            "size": self.model.geom_size[gid].copy(),
            "group": int(self.model.geom_group[gid]),
        })
        self.model.geom_pos[gid] = snap["pos"]
        self.model.geom_quat[gid] = snap["quat"]
        self.model.geom_size[gid] = snap["size"]
        self.model.geom_group[gid] = snap["group"]
        if self.cur_state and self.cur_state.gid == gid:
            self.cur_state = self._load_state(self.cur_state.bid, gid)

    # ── Sync model ←→ state ─────────────────────────────────────────────────

    def _sync_rbound(self, gid: int):
        m = self.model
        t, sz = m.geom_type[gid], m.geom_size[gid]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            m.geom_rbound[gid] = float(sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            m.geom_rbound[gid] = float(np.linalg.norm(sz[:3]))
        else:
            m.geom_rbound[gid] = float(np.hypot(sz[0], sz[1]))

    def _apply_state_to_model(self):
        """Write cur_state fields into the MuJoCo model arrays."""
        s = self.cur_state
        if s is None:
            return
        m = self.model
        m.geom_pos[s.gid] = s.pos
        m.geom_quat[s.gid] = s.quat
        m.geom_size[s.gid][:len(s.size)] = s.size[:len(s.size)]
        self._sync_rbound(s.gid)

    # ── Viewer highlight ────────────────────────────────────────────────────

    def _sync_selection(self, handle):
        """Two-way link between GUI body selection and the viewer's native
        double-click body-select highlight (mjVIS_SELECT + perturb.select).

        Reads first (native double-click may have changed perturb.select
        since last frame) so a viewer click updates the GUI; writes last so
        a GUI list-pick forces the viewer's highlight to follow. Only
        `select` is touched — `active` stays untouched, so no perturbation
        force gets applied.

        perturb.select never resets after a double-click, so it must be
        compared against the last *seen* value (edge-triggered), not the
        current GUI selection — comparing to the GUI selection would treat
        the same stale click as "new" on every frame after a GUI-side pick,
        snapping the selection back to whatever was last clicked natively.
        """
        clicked_bid = int(handle.perturb.select)
        if clicked_bid != self._last_perturb_select:
            self._last_perturb_select = clicked_bid
            if clicked_bid >= 0:
                name = self.model.body(clicked_bid).name
                if name in self.body_names:
                    slots = self._slot_gids(clicked_bid)
                    jid = self._body_jid(clicked_bid)
                    with self.lock:
                        self.selected_body = name
                        self.selected_slot_idx = 0
                        self.cur_state = self._load_state(clicked_bid, slots[0]) if slots else None
                        self.cur_jid = jid
                    n_enabled = sum(1 for g in slots if self.model.geom_group[g] == self.collision_group)
                    dpg.set_value("body_list", name)
                    dpg.set_value("body_label", f"body: {name}  ({n_enabled} collision geom(s))")
                    dpg.set_value("mirror_target", self._auto_mirror_target())
                    self._rebuild_pose_widgets()
                    self._refresh_joint_widgets()
        cur_bid = self._bid(self.selected_body)
        handle.perturb.select = cur_bid
        self._last_perturb_select = cur_bid

    def _update_highlight(self, handle):
        """Draw body-frame axes at the selected body origin."""
        bid = self._bid(self.selected_body)
        origin = self.data.xpos[bid]
        xmat = self.data.xmat[bid].reshape(3, 3)
        handle.user_scn.ngeom = 0
        for i in range(3):
            axis_dir = xmat[:, i]
            g = handle.user_scn.geoms[handle.user_scn.ngeom]
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                origin, origin + 0.1 * axis_dir,
            )
            g.rgba[:] = _AXIS_COLORS[i]
            handle.user_scn.ngeom += 1

    # ── GUI (dearpygui) ─────────────────────────────────────────────────────

    def _build_gui(self):
        dpg.create_context()
        with dpg.window(label=f"Collision Editor: {self.xml_basename}",
                        tag="main", width=640, height=820):
            dpg.add_text("Select body:")
            dpg.add_listbox(self.body_names, tag="body_list",
                            default_value=self.body_names[0] if self.body_names else "",
                            num_items=10, callback=self._on_select_body)
            dpg.add_text(f"body: {self.selected_body}", tag="body_label")
            with dpg.group(tag="group_joint_angle", show=False):
                dpg.add_text("Joint angle:", tag="joint_name_text")
                dpg.add_drag_float(label="", tag="joint_angle_slider", speed=0.5,
                                   min_value=-180.0, max_value=180.0,
                                   callback=lambda s, a: self._on_joint_angle(a))
            dpg.add_separator()
            dpg.add_text("Collision-geom slots (per body):")
            dpg.add_listbox([], tag="slot_list", num_items=8,
                            callback=self._on_select_slot)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add geom (next empty slot)",
                               callback=lambda: self._add_geom())
                dpg.add_button(label="Remove selected",
                               callback=lambda: self._remove_geom())
            with dpg.group(horizontal=True):
                dpg.add_combo(self.body_names, tag="mirror_target", width=200,
                             default_value=self._auto_mirror_target())
                dpg.add_combo(["mirror (flip Y)", "copy exact"], tag="mirror_mode",
                             width=140, default_value="mirror (flip Y)")
                dpg.add_button(label="Mirror -> target",
                               callback=lambda: self._do_mirror())
            dpg.add_separator()

            dpg.add_checkbox(label="Enabled (visible in group 3)",
                             tag="enabled_cb", default_value=False,
                             callback=self._on_enabled)
            dpg.add_text("type: —", tag="type_text")

            # Common: position + radius.
            with dpg.group(tag="group_pose_common", show=False):
                dpg.add_drag_float(label="pos x", tag="px", speed=0.001,
                                   min_value=-1.0, max_value=1.0,
                                   callback=lambda s, a: self._on_pos(0, a))
                dpg.add_drag_float(label="pos y", tag="py", speed=0.001,
                                   min_value=-1.0, max_value=1.0,
                                   callback=lambda s, a: self._on_pos(1, a))
                dpg.add_drag_float(label="pos z", tag="pz", speed=0.001,
                                   min_value=-1.0, max_value=1.0,
                                   callback=lambda s, a: self._on_pos(2, a))
                dpg.add_drag_float(label="size[0] / radius", tag="s0",
                                   speed=0.001, min_value=0.002, max_value=0.3,
                                   callback=lambda s, a: self._on_size(0, a))

            # Cylinder/capsule: halflen + rotation.
            with dpg.group(tag="group_cylcap", show=False):
                dpg.add_drag_float(label="size[1] / halflen", tag="s1",
                                   speed=0.001, min_value=0.002, max_value=0.3,
                                   callback=lambda s, a: self._on_size(1, a))

            # Box: extra half-extents.
            with dpg.group(tag="group_box", show=False):
                dpg.add_drag_float(label="size[1] / half-y", tag="s1b",
                                   speed=0.001, min_value=0.002, max_value=0.3,
                                   callback=lambda s, a: self._on_size(1, a))
                dpg.add_drag_float(label="size[2] / half-z", tag="s2b",
                                   speed=0.001, min_value=0.002, max_value=0.3,
                                   callback=lambda s, a: self._on_size(2, a))

            # Rotation (all non-sphere types).
            with dpg.group(tag="group_rotation", show=False):
                dpg.add_drag_float(label="rot x (deg)", tag="rx", speed=0.5,
                                   min_value=-180, max_value=180,
                                   callback=lambda s, a: self._on_euler(0, a))
                dpg.add_drag_float(label="rot y (deg)", tag="ry", speed=0.5,
                                   min_value=-180, max_value=180,
                                   callback=lambda s, a: self._on_euler(1, a))
                dpg.add_drag_float(label="rot z (deg)", tag="rz", speed=0.5,
                                   min_value=-180, max_value=180,
                                   callback=lambda s, a: self._on_euler(2, a))
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Snap: joint axis",
                                   callback=lambda: self._snap_to(self._joint_axis_local))
                    dpg.add_button(label="Snap: bone axis",
                                   callback=lambda: self._snap_to(self._bone_axis_local))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Auto-fit: joint axis",
                                   callback=lambda: self._auto_fit(self._joint_axis_local))
                    dpg.add_button(label="Auto-fit: bone axis",
                                   callback=lambda: self._auto_fit(self._bone_axis_local))

            dpg.add_separator()
            dpg.add_text("(geom XML preview)", tag="line_text", wrap=600)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Undo", callback=lambda: self._do_undo())
                dpg.add_button(label="Redo", callback=lambda: self._do_redo())
            dpg.add_button(label="Print ALL to console + clipboard",
                           callback=lambda: self._do_print_all())
            default_save_path = str(pathlib.Path(self.xml_path).with_stem(
                pathlib.Path(self.xml_path).stem + "_edited"))
            dpg.add_input_text(label="save path", tag="save_path",
                               default_value=default_save_path, width=420)
            with dpg.group(horizontal=True):
                if self.write_xml:
                    dpg.add_button(label="Save (overwrite source)",
                                   callback=lambda: self._do_write_xml(self.xml_path))
                dpg.add_button(label="Save As",
                               callback=lambda: self._do_write_xml(dpg.get_value("save_path")))

        dpg.create_viewport(title=f"Collision Editor - {self.xml_basename}",
                            width=660, height=860)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _refresh_slot_list(self):
        bid = self._bid(self.selected_body)
        labels = []
        for i, gid in enumerate(self._slot_gids(bid)):
            m = self.model
            type_name = _GEOM_TYPE_NAME.get(m.geom_type[gid], "?")
            status = "ON" if m.geom_group[gid] == self.collision_group else "off"
            labels.append(f"slot {i}: {type_name} [{status}]")
        dpg.configure_item("slot_list", items=labels)
        if self.selected_slot_idx < len(labels):
            dpg.set_value("slot_list", labels[self.selected_slot_idx])

    def _rebuild_pose_widgets(self):
        s = self.cur_state
        if s is None:
            dpg.configure_item("group_pose_common", show=False)
            dpg.configure_item("group_cylcap", show=False)
            dpg.configure_item("group_box", show=False)
            dpg.configure_item("group_rotation", show=False)
            return

        is_sphere = s.geom_type == "sphere"
        is_cylcap = s.geom_type in ("cylinder", "capsule")
        is_box = s.geom_type == "box"

        dpg.configure_item("group_pose_common", show=s.enabled)
        dpg.configure_item("group_cylcap", show=is_cylcap and s.enabled)
        dpg.configure_item("group_box", show=is_box and s.enabled)
        dpg.configure_item("group_rotation", show=(not is_sphere) and s.enabled)

        dpg.set_value("enabled_cb", s.enabled)
        dpg.set_value("type_text", f"type: {s.geom_type}")
        dpg.set_value("px", float(s.pos[0]))
        dpg.set_value("py", float(s.pos[1]))
        dpg.set_value("pz", float(s.pos[2]))
        dpg.set_value("s0", float(s.size[0]))

        if is_cylcap:
            dpg.set_value("s1", float(s.size[1]))
        if is_box:
            dpg.set_value("s1b", float(s.size[1]))
            dpg.set_value("s2b", float(s.size[2]))
        if not is_sphere:
            dpg.set_value("rx", float(s.euler_deg[0]))
            dpg.set_value("ry", float(s.euler_deg[1]))
            dpg.set_value("rz", float(s.euler_deg[2]))

        xml_line = self._geom_xml_line(s.gid) or "(disabled)"
        dpg.set_value("line_text", xml_line)
        self._refresh_slot_list()

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_select_body(self, sender, name):
        bid = self._bid(name)
        slots = self._slot_gids(bid)
        jid = self._body_jid(bid)
        with self.lock:
            self.selected_body = name
            self.selected_slot_idx = 0
            if slots:
                self.cur_state = self._load_state(bid, slots[0])
            else:
                self.cur_state = None
            self.cur_jid = jid
        n_enabled = sum(1 for g in slots if self.model.geom_group[g] == self.collision_group)
        dpg.set_value("body_label", f"body: {name}  ({n_enabled} collision geom(s))")
        dpg.set_value("mirror_target", self._auto_mirror_target())
        self._rebuild_pose_widgets()
        self._refresh_joint_widgets()

    def _on_select_slot(self, sender, label):
        idx = int(label.split()[1].rstrip(":"))
        bid = self._bid(self.selected_body)
        slots = self._slot_gids(bid)
        with self.lock:
            self.selected_slot_idx = idx
            self.cur_state = self._load_state(bid, slots[idx])
        self._rebuild_pose_widgets()

    def _on_enabled(self, sender, val):
        if self.cur_state is None:
            return
        self._push_undo()
        with self.lock:
            self.cur_state.enabled = val
            self.model.geom_group[self.cur_state.gid] = (
                self.collision_group if val else self.disabled_group
            )
        self._rebuild_pose_widgets()

    def _on_pos(self, i: int, val: float):
        if self.cur_state is None:
            return
        with self.lock:
            self.cur_state.pos[i] = val
            self.model.geom_pos[self.cur_state.gid][i] = val
        dpg.set_value("line_text", self._geom_xml_line(self.cur_state.gid) or "(disabled)")

    def _on_joint_angle(self, val: float):
        """Pose the robot by driving qpos directly (not a saved edit — lets
        you swing a joint through its range to check collision-geom
        coverage against the mesh at non-zero poses)."""
        if self.cur_jid is None:
            return
        m = self.model
        qadr = m.jnt_qposadr[self.cur_jid]
        is_hinge = m.jnt_type[self.cur_jid] == mujoco.mjtJoint.mjJNT_HINGE
        with self.lock:
            self.data.qpos[qadr] = np.deg2rad(val) if is_hinge else val

    def _refresh_joint_widgets(self):
        if self.cur_jid is None:
            dpg.configure_item("group_joint_angle", show=False)
            return
        dpg.configure_item("group_joint_angle", show=True)
        m = self.model
        jid = self.cur_jid
        name = m.joint(jid).name
        qadr = m.jnt_qposadr[jid]
        is_hinge = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        if m.jnt_limited[jid]:
            lo, hi = m.jnt_range[jid]
        else:
            lo, hi = (-np.pi, np.pi) if is_hinge else (-0.5, 0.5)
        lo_disp = np.rad2deg(lo) if is_hinge else lo
        hi_disp = np.rad2deg(hi) if is_hinge else hi
        unit = "deg" if is_hinge else "m"
        dpg.configure_item("joint_angle_slider", label=f"{unit}",
                           min_value=float(lo_disp), max_value=float(hi_disp))
        dpg.set_value("joint_name_text", f"Joint angle: {name}")
        cur = self.data.qpos[qadr]
        dpg.set_value("joint_angle_slider", float(np.rad2deg(cur)) if is_hinge else float(cur))

    def _on_euler(self, i: int, val: float):
        if self.cur_state is None:
            return
        with self.lock:
            s = self.cur_state
            delta = val - s.euler_deg[i]
            s.euler_deg[i] = val
            axis = np.zeros(3)
            axis[i] = 1.0
            half = np.deg2rad(delta) / 2
            dq = np.array([np.cos(half), *(axis * np.sin(half))])
            s.quat = _quat_mul(s.quat, dq)
            self.model.geom_quat[s.gid] = s.quat
        dpg.set_value("line_text", self._geom_xml_line(s.gid) or "(disabled)")

    def _on_size(self, i: int, val: float):
        if self.cur_state is None:
            return
        with self.lock:
            s = self.cur_state
            s.size[i] = val
            self.model.geom_size[s.gid][i] = val
            self._sync_rbound(s.gid)
        dpg.set_value("line_text", self._geom_xml_line(s.gid) or "(disabled)")

    def _snap_to(self, axis_fn):
        s = self.cur_state
        if s is None:
            return
        axis = axis_fn(s.bid)
        if axis is None:
            return
        self._push_undo()
        q = _quat_from_z_to(axis)
        e_deg = np.rad2deg(_quat_to_euler_xyz(q))
        with self.lock:
            s.quat = q
            s.euler_deg = e_deg
            self.model.geom_quat[s.gid] = q
        dpg.set_value("rx", e_deg[0])
        dpg.set_value("ry", e_deg[1])
        dpg.set_value("rz", e_deg[2])
        dpg.set_value("line_text", self._geom_xml_line(s.gid) or "(disabled)")

    def _auto_fit(self, axis_fn):
        s = self.cur_state
        if s is None or s.geom_type == "sphere":
            return
        axis = axis_fn(s.bid)
        mesh_path = self._mesh_path_for_body(s.bid)
        if axis is None or mesh_path is None:
            return
        self._push_undo()
        radius, halflen, pos, quat = _fit_from_mesh(mesh_path, axis)
        e_deg = np.rad2deg(_quat_to_euler_xyz(quat))
        with self.lock:
            s.pos = pos
            s.quat = quat
            s.euler_deg = e_deg
            s.size = np.array([radius, halflen, 0.0])
            self.model.geom_pos[s.gid] = pos
            self.model.geom_quat[s.gid] = quat
            self.model.geom_size[s.gid] = s.size
            self._sync_rbound(s.gid)
        dpg.set_value("px", pos[0])
        dpg.set_value("py", pos[1])
        dpg.set_value("pz", pos[2])
        dpg.set_value("s0", radius)
        dpg.set_value("s1", halflen)
        dpg.set_value("rx", e_deg[0])
        dpg.set_value("ry", e_deg[1])
        dpg.set_value("rz", e_deg[2])
        dpg.set_value("line_text", self._geom_xml_line(s.gid) or "(disabled)")

    def _add_geom(self):
        """Enable the first disabled slot on the current body."""
        bid = self._bid(self.selected_body)
        slots = self._slot_gids(bid)
        for i, gid in enumerate(slots):
            if self.model.geom_group[gid] == self.disabled_group:
                self._push_undo()
                with self.lock:
                    self.selected_slot_idx = i
                    self.model.geom_group[gid] = self.collision_group
                    self.cur_state = self._load_state(bid, gid)
                self._rebuild_pose_widgets()
                return
        print(f"{self.selected_body}: all {len(slots)} slots in use")

    def _remove_geom(self):
        """Disable the currently selected slot."""
        self._on_enabled(None, False)

    def _auto_mirror_target(self) -> str:
        """Auto-detected left/right counterpart of selected_body, if it exists."""
        mirrored = _mirror_body_name(self.selected_body)
        return mirrored if mirrored in self.body_names else ""

    def _mirror_to_other_side(self, target_name: str, flip: bool = True):
        """Overwrite *target_name*'s collision slots with a copy of
        selected_body's enabled slots, matched by geom type (slot count fixed
        at build time, so type match is exact-in/exact-out, no reflow).

        flip=True reflects pos/quat across the local XZ-plane (for sibling
        bodies sharing one axis convention, see _mirror_pos_quat). flip=False
        copies pos/quat unchanged (for robots whose left/right body frames
        are themselves already mirrored, e.g. via body_quat, so the raw
        local-frame values already line up)."""
        src_bid = self._bid(self.selected_body)
        dst_bid = self._bid(target_name)
        if dst_bid < 0:
            print(f"mirror: body '{target_name}' not found")
            return
        if dst_bid == src_bid:
            print("mirror: target same as source")
            return
        m = self.model
        src_by_type: dict[str, list[int]] = {t: [] for t in _SLOT_TYPES}
        for gid in self._slot_gids(src_bid):
            if m.geom_group[gid] == self.collision_group:
                src_by_type[_GEOM_TYPE_NAME[m.geom_type[gid]]].append(gid)
        dst_by_type: dict[str, list[int]] = {t: [] for t in _SLOT_TYPES}
        for gid in self._slot_gids(dst_bid):
            dst_by_type[_GEOM_TYPE_NAME[m.geom_type[gid]]].append(gid)

        with self.lock:
            for gid in self._slot_gids(dst_bid):
                self._push_undo_gid(gid)
                m.geom_group[gid] = self.disabled_group
            for type_name, src_gids in src_by_type.items():
                dst_gids = dst_by_type[type_name]
                if len(src_gids) > len(dst_gids):
                    print(f"mirror: {target_name} has only {len(dst_gids)} "
                          f"{type_name} slot(s), dropping {len(src_gids) - len(dst_gids)}")
                for src_gid, dst_gid in zip(src_gids, dst_gids):
                    if flip:
                        pos, quat = _mirror_pos_quat(m.geom_pos[src_gid], m.geom_quat[src_gid])
                    else:
                        pos, quat = m.geom_pos[src_gid].copy(), m.geom_quat[src_gid].copy()
                    m.geom_pos[dst_gid] = pos
                    m.geom_quat[dst_gid] = quat
                    m.geom_size[dst_gid] = m.geom_size[src_gid].copy()
                    m.geom_group[dst_gid] = self.collision_group
                    self._sync_rbound(dst_gid)

        if self.cur_state is not None and self.cur_state.bid == dst_bid:
            self.cur_state = self._load_state(dst_bid, self.cur_state.gid)
        print(f"mirrored {self.selected_body} -> {target_name}")

    def _do_mirror(self):
        target = dpg.get_value("mirror_target")
        if not target:
            print("mirror: no target body selected")
            return
        flip = dpg.get_value("mirror_mode") == "mirror (flip Y)"
        self._mirror_to_other_side(target, flip=flip)
        self._rebuild_pose_widgets()

    def _do_undo(self):
        with self.lock:
            self._undo()
        self._rebuild_pose_widgets()

    def _do_redo(self):
        with self.lock:
            self._redo()
        self._rebuild_pose_widgets()

    def _do_print_all(self):
        text = self.print_all()
        try:
            dpg.set_clipboard_text(text)
        except Exception:
            pass  # Clipboard not available in headless.

    def _do_write_xml(self, path: str):
        """Write current collision geoms into a copy of the source XML at *path*."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        def _update_body(body_el: ET.Element):
            body_name = body_el.get("name", "")
            bid = None
            try:
                bid = self._bid(body_name)
            except Exception:
                pass

            if bid is not None and body_name in self.body_names:
                # Remove existing collision geoms.
                to_remove = []
                for g in body_el.findall("geom"):
                    cls = g.get("class", "")
                    grp = g.get("group", "")
                    if cls == "collision" or grp == str(self.collision_group):
                        to_remove.append(g)
                for g in to_remove:
                    body_el.remove(g)
                # Add current enabled slots.
                for gid in self._slot_gids(bid):
                    xl = self._geom_xml_line(gid)
                    if xl is not None:
                        new_g = ET.fromstring(xl)
                        body_el.append(new_g)

            for child in body_el.findall("body"):
                _update_body(child)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for top in worldbody.findall("body"):
                _update_body(top)

        tree.write(path, xml_declaration=True)
        print(f"wrote collision geoms to {path}")

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        """Launch GUI thread + MuJoCo viewer on main thread."""
        if not _HAS_DPG:
            raise RuntimeError("dearpygui not installed: pip install dearpygui")

        gui_ready = threading.Event()

        def _gui_thread():
            self._build_gui()
            gui_ready.set()
            dpg.start_dearpygui()
            dpg.destroy_context()

        threading.Thread(target=_gui_thread, daemon=True).start()
        gui_ready.wait()
        self._rebuild_pose_widgets()
        self._refresh_joint_widgets()

        # Make visual meshes translucent so collision is visible.
        for gid in range(self.model.ngeom):
            if self.model.geom_group[gid] in self.visual_groups:
                self.model.geom_rgba[gid][3] = 0.3

        with viewer.launch_passive(self.model, self.data) as handle:
            handle.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = True
            # Show all groups except disabled.
            for g_idx in range(6):
                handle.opt.geomgroup[g_idx] = 0 if g_idx == self.disabled_group else 1
            while handle.is_running() and dpg.is_dearpygui_running():
                with self.lock:
                    self._apply_state_to_model()
                mujoco.mj_forward(self.model, self.data)
                self._sync_selection(handle)
                self._update_highlight(handle)
                handle.sync()

        # Final dump on exit.
        self.print_all()

        # Clean up temp file.
        try:
            os.unlink(self._aug_xml.name)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generic per-link collision-geom editor for any MuJoCo MJCF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--xml", type=str, required=True,
                        help="Path to MuJoCo MJCF XML file.")
    parser.add_argument("--assets", type=str, default=None,
                        help="Path to mesh assets directory (auto-detected if omitted).")
    parser.add_argument("--visual-groups", type=str, default="1,2",
                        help="Comma-separated visual geom groups (default: 1,2).")
    parser.add_argument("--collision-group", type=int, default=3,
                        help="Collision geom group (default: 3).")
    parser.add_argument("--disabled-group", type=int, default=5,
                        help="Group for hidden placeholder geoms (default: 5).")
    parser.add_argument("--write-xml", action="store_true",
                        help="Enable 'Write to source XML' button in the GUI.")
    args = parser.parse_args()

    visual_groups = {int(x) for x in args.visual_groups.split(",")}

    editor = CollisionEditor(
        xml_path=args.xml,
        assets_dir=args.assets,
        visual_groups=visual_groups,
        collision_group=args.collision_group,
        disabled_group=args.disabled_group,
        write_xml=args.write_xml,
    )
    editor.run()


if __name__ == "__main__":
    main()
