"""MuJoCo -> Blender live photoreal bridge (MuJoCo side).

Why
---
MuJoCo's renderer is fixed-function Phong: no global illumination, no ambient occlusion,
no PBR materials, no real sky. Relighting it (see ``mj_envs/visual_quality.py``) makes it
*legible* but cannot make it photoreal — that is a property of the renderer, not the scene.
Blender's EEVEE Next does raytraced GI, reflections and soft shadows at interactive rates on
an RTX 4090, and Cycles/OptiX is available for stills.

Design
------
Blender is a **pure render slave**. MuJoCo remains the single source of truth for physics, so
what you watch is exactly the rollout the policy produced. Two channels:

* **Static, once at setup**: geometry, materials and mesh/heightfield data are written to an
  ``.npz`` that the Blender-side script loads and turns into real objects.
* **Live, per frame**: a Unix datagram of ``n_dyn * 12`` float32 — world position (3) and the
  row-major rotation matrix (9) for each geom that can *move* (see ``dynamic_geom_mask``; the
  static ones are already placed by the export and never change) — followed by ``nlight * 6``:
  world position (3) and direction (3) per light, for the lights that track the robot — followed
  by a final 3: the camera pivot, i.e. MuJoCo's tracking-camera ``lookat``. On a scene shaped
  like Argus/Mars that is 41 of 216 geoms, 2 kB per frame, far under the 64 kB datagram limit.

Rejected alternatives
---------------------
* **Genesis RayTracer** — needs the unfetched LuisaRender submodule plus a CUDA toolkit and
  clang that are not installed, and it would re-simulate under a different contact model than
  the policy trained against.
* **Isaac Sim** — ~25 GB on a disk sitting at 95% full.
* **``mujoco.usd`` export** — no ``mjGEOM_HFIELD`` support (``mujoco/usd/shapes.py`` handles
  plane/sphere/capsule/ellipsoid/cylinder/box/mesh only), so the terrain would vanish. Writing
  the heightfield path is needed either way, at which point the USD hop only adds a format.
* **A stream socket instead of datagrams** — a dropped render frame is not worth a
  head-of-line stall in the physics loop.

Invariant
---------
The Blender side indexes purely by position in ``vis_ids``. If the geom filter here and the
object build order there ever disagree, every object is driven by the wrong geom's pose. The
order is "ascending geom id, filtered by visible group" in both places and must stay that way.

The same applies one level down: the pose datagram is indexed by position within
``dynamic_geom_mask``'s subset, which is shipped in the scene payload as ``geom_dynamic`` so both
sides read one array rather than each deriving it.
"""

from __future__ import annotations

import os
import shutil
import socket
import struct
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from mjlab.viewer.base import BaseViewer
from mjlab.viewer.offscreen_renderer import _get_camera_body_id
from mjlab.viewer.viewer_config import ViewerConfig

from asset_zoo.fov_frustum import FOV_GEOM_GROUP

_SCENE_DIR = Path("/tmp/mj_photoreal")
_SCENE_PATH = _SCENE_DIR / "scene.npz"
_BLENDER_SCRIPT = Path(__file__).parent / "blender_view.py"
_ROUTE_MAGIC = b"RTE1"
_ROUTE_HEADER = struct.Struct("<4sII")
_MAX_PACKET = 65535
_UNCHANGED = object()

# Groups 0-2 are MuJoCo's visual convention (the native viewer default). FOV_GEOM_GROUP (4) is
# the head-camera FOV overlay -- translucent emissive frustum meshes baked onto each camera's
# gimbal link by asset_zoo.fov_frustum.add_fov_frustum_hull. They are ordinary geoms rigidly
# fixed to the moving link, so streaming them through the normal geom-pose path makes the FOV
# track the gimbal for free -- no separate camera stream, no frustum built Blender-side. The
# native viewer hides group 4 by default (digit-4 toggle); the photoreal viewer shows it because
# visualising the FOV is the whole point of turning this bridge on for a camera robot.
_VISIBLE_GROUPS = (0, 1, 2, FOV_GEOM_GROUP)

# MuJoCo's compile-time ``geom_rgba`` default. A compiled model cannot report whether an rgba was
# authored or left alone, so this value is the test -- the same one ``blender_view._geom_material``
# uses. See ``_resolved_rgba``.
_MJ_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

# ``PoseReceiver``'s modal timer in blender_view.py. The play loop defaults to frame_rate=999
# ("fast as possible", right for the native viewer whose draw *is* the frame), but Blender only
# drains the socket at this rate, so every packet beyond it is built and thrown away. Measured on
# ArgusMini20VelMarsMeshTerrain: 999 Hz x 0.25 ms of pull+pack+send is ~25% of the tick loop
# spent on frames nobody sees. Keep in sync with the timer period there.
_BLENDER_TIMER_HZ = 60.0

# Default render resolution. Callers that also write a ``mujoco.Renderer`` video of the same
# rollout pass their own so the two are frame-for-frame comparable.
_RENDER_SIZE = (1280, 720)

# Concurrent headless replay processes one *idle* recorder may split a clip across. The curve is
# strongly sublinear because the shards share one GPU and one OpenGL scheduler: measured on 240
# frames at 720p, speedup is 1.00 / 1.27 / 1.33 / 1.56 / 1.62 for 1 / 2 / 3 / 4 / 6 processes and
# turns negative by 8. Four is where the return per process stops paying for the memory of another
# Blender holding the whole scene.
_MAX_REPLAY_PROCS = 4
# Minimum frames per shard. A shard costs ~1.9 s of process start plus scene load (measured with
# a warm .blend cache) against ~85 ms a frame at 720p, so it breaks even around 22 frames; 60
# keeps a comfortable margin over that on the slowest scenes.
_SHARD_FRAMES = 60
_INFLIGHT: list[subprocess.Popen] = []


def _claim_shards(nframes: int) -> int:
  """Processes this clip may split across, given what is already rendering.

  This caps *sharding*, not total concurrency: every recorder always gets its own process, so a
  caller that records six cameras off one rollout (``joint_monkey_grid``) still starts six
  Blenders. What the cap prevents is those six each multiplying themselves.
  """
  _INFLIGHT[:] = [p for p in _INFLIGHT if p.poll() is None]
  return max(1, min(nframes // _SHARD_FRAMES, _MAX_REPLAY_PROCS - len(_INFLIGHT)))


def visible_geom_ids(model, geomgroup=None) -> np.ndarray:
  """Geom ids the bridge renders, ascending. Shared ordering contract with the Blender side.

  ``geomgroup`` is a caller's ``MjvOption.geomgroup``, honoured verbatim so a Blender render shows
  exactly what that caller's MuJoCo render shows. Without it the FOV hull (group 4) is always on,
  which is right for a play session opened to look at the FOV and wrong for the reach recorder,
  where ``apply_view_options`` hides it: its 5 m emissive shells then fill most of the frame.
  ``None`` keeps the play default.
  """
  groups = np.flatnonzero(np.asarray(geomgroup)) if geomgroup is not None else _VISIBLE_GROUPS
  keep = np.isin(model.geom_group, groups)
  # Fully transparent geoms contribute no pixels, and MuJoCo's own renderer skips them too, so
  # dropping them changes nothing visible. It is not a micro-optimisation here: the Mars/Moon
  # terrains hide a collision heightfield behind the visual mesh at alpha 0 (see srb_baked_mesh),
  # and shipping it meant a 41 MB array in the scene payload that Blender then expanded into a
  # 3200x3200 grid -- 10.2 M vertices and 20.5 M triangles built to be invisible.
  return np.flatnonzero(keep & (_resolved_rgba(model)[:, 3] > 0)).astype(np.int32)


def _resolved_rgba(model) -> np.ndarray:
  """The RGBA MuJoCo actually draws each geom with, per geom id.

  Probed off ``mjv_updateScene``: a geom whose ``geom_rgba`` differs from the compile-time default
  overrides its material **outright -- colour and alpha alike**; only a geom left at the default
  takes the material's. Taking the material's alpha unconditionally, as this used to, silently
  deleted any geom that made a transparent material opaque again (``mat_rgba`` alpha 0,
  ``geom_rgba`` alpha 1): the filter dropped it from ``vis_ids``, so it was never built, while
  MuJoCo drew it solid.

  The two exceptions mirror ``blender_view._geom_material``, which resolves the *material* for the
  same geom on the other side of the wire and must agree with this: an emissive or textured
  material wins regardless, because a flat rgba can express neither a glow nor a map.
  """
  rgba = model.geom_rgba.copy()
  matid = model.geom_matid
  has_mat = matid >= 0
  authored = ~np.all(np.isclose(model.geom_rgba, _MJ_DEFAULT_RGBA), axis=1)
  plain = np.zeros(len(rgba), dtype=bool)
  plain[has_mat] = (model.mat_texid[matid[has_mat]] < 0).all(axis=1) & (
    model.mat_emission[matid[has_mat]] == 0.0
  )
  take_mat = has_mat & ~(authored & plain)
  rgba[take_mat] = model.mat_rgba[matid[take_mat]]
  return rgba


def dynamic_geom_mask(model, vis: np.ndarray) -> np.ndarray:
  """Which of ``vis`` can move, as a bool mask in ``vis`` order.

  A geom on a body welded to the world with no mocap has a ``geom_xpos``/``geom_xmat`` that
  ``mj_kinematics`` computes identically for every ``qpos``. It is therefore not a heuristic that
  it never moves, it is a property of the compiled model -- so the build-time pose Blender already
  has is its pose for the whole session and streaming it 60x a second is pure waste. The test is
  the same one ``track_body_id`` uses to find the floating base, inverted.

  Terrain is where this pays: on a scene shaped like ArgusMini's Mars (174 static props, 40 robot
  links) it drops the datagram from 10404 to 2020 bytes, the recorded pose file 5x, and the
  per-frame Blender-side matrix write from 3.10 ms to 0.39 ms -- 16% of a 60 Hz budget.

  Not covered, deliberately: a caller that mutates ``model.body_pos``/``geom_pos`` after
  ``export_scene``. Nothing here does, and the alternative is streaming every geom forever.
  """
  bodies = model.geom_bodyid[vis]
  welded = model.body_weldid[bodies] == 0
  mocap = model.body_mocapid[model.body_rootid[bodies]] >= 0
  return ~welded | mocap


def track_body_id(model, cfg: ViewerConfig, entities) -> int:
  """Body the camera centres on, resolved exactly as ``NativeMujocoViewer`` resolves it.

  ``_get_camera_body_id`` covers ASSET_ROOT/ASSET_BODY and returns -1 for AUTO and WORLD, but
  the *native* viewer does not leave AUTO free: it scans for the first body that is not welded
  to the world, which is the robot's floating base. That scan is reproduced here so the
  photoreal camera tracks the same body as the native one for the same ``ViewerConfig``.

  Returns -1 for WORLD (and for a model with no free body), meaning "no tracking": the pivot is
  then the static ``cfg.lookat``, again as MuJoCo does.
  """
  if cfg.origin_type is not ViewerConfig.OriginType.AUTO:
    return _get_camera_body_id(cfg, entities)
  for b in range(model.nbody):
    if not (model.body_weldid[b] == 0 and model.body_mocapid[model.body_rootid[b]] < 0):
      return b
  return -1


def _lookat(data, cfg: ViewerConfig, track_bid: int) -> np.ndarray:
  """Point the camera orbits, following MuJoCo's tracking camera.

  MuJoCo centres a tracking camera on ``subtree_com[trackbodyid]``, but mjlab's native viewer
  overwrites that entry with the body's frame origin every sync (``_stabilize_tracking_camera``)
  so domain-randomised inertials cannot drift the view. ``xpos`` is therefore what the native
  viewer actually looks at, and it is also steadier than the visible-geom centroid this replaces,
  which swung with every limb.

  No smoothing: ``mjv_updateCamera`` copies the tracked point straight into ``lookat`` (probed --
  a 10 m teleport lands in one frame), so a filter here would not be "the same logic".
  """
  if track_bid < 0:
    return np.asarray(cfg.lookat, dtype=np.float32)
  return data.xpos[track_bid].astype(np.float32)


def export_scene(
  model,
  data,
  cfg: ViewerConfig,
  track_bid: int,
  path: Path = _SCENE_PATH,
  *,
  geomgroup=None,
  mujoco_look: bool = False,
  size: tuple[int, int] = _RENDER_SIZE,
) -> Path:
  """Write the static scene description Blender needs to build the objects once.

  Args:
      model: compiled ``mujoco.MjModel``.
      data: ``mujoco.MjData`` already advanced by ``mj_forward``, used for the rest pose so a
          headless still render works without a live stream.
      cfg: the env's ``ViewerConfig``, source of the camera framing.
      track_bid: body the camera tracks, from ``track_body_id``; -1 for a static pivot.
      path: destination ``.npz``.
      geomgroup: caller's ``MjvOption.geomgroup``; see ``visible_geom_ids``.
      mujoco_look: reproduce MuJoCo's *light levels* -- headlight, unscaled ambient, no filmic
          tonemap -- instead of the photoreal grade. Shading stays EEVEE's (GI, soft shadows, PBR
          materials); only the rig it is lit by changes. On for anything whose output is compared
          against a ``mujoco.Renderer`` video of the same rollout.
      size: render resolution, matched to the caller's own recorder so the two videos are
          interchangeable.

  Returns:
      ``path``, for convenience.
  """
  vis = visible_geom_ids(model, geomgroup)
  out: dict[str, np.ndarray] = {
    "vis_ids": vis,
    "geom_type": model.geom_type[vis],
    "geom_size": model.geom_size[vis],
    "geom_rgba": model.geom_rgba[vis],
    "geom_matid": model.geom_matid[vis],
    "geom_dataid": model.geom_dataid[vis],
    # The FOV hull is a translucent shell, but a shadow pass has no notion of alpha: left to
    # cast, it drops an opaque slab of cone-shaped darkness across the floor. MuJoCo has no
    # per-geom shadow opt-out, which is why run.py kills castshadow on every light for the native
    # viewer; Blender does have one (``Object.visible_shadow``), so only these geoms are excluded
    # and the robot keeps its ground shadow. Shipped as a mask rather than the group id so
    # FOV_GEOM_GROUP stays defined in exactly one place (the `mujoco` package, and therefore this
    # constant, is not importable inside Blender).
    "geom_noshadow": model.geom_group[vis] == FOV_GEOM_GROUP,
    # Which objects the per-frame datagram will actually carry; see ``dynamic_geom_mask``. The
    # rest are driven once, from geom_pos0/geom_mat0 below, and then left alone.
    "geom_dynamic": dynamic_geom_mask(model, vis),
    "geom_pos0": data.geom_xpos[vis].astype(np.float32),
    "geom_mat0": data.geom_xmat[vis].astype(np.float32),
    # Material channels. MuJoCo has no metallic term, so the Blender side derives one from
    # specular; see blender_view.py for that mapping and why. mat_emission drives the FOV hull
    # overlay: those materials bake emission=1 so the translucent frustum glows against the dark
    # world instead of washing out to invisible at its alpha 0.01.
    "mat_rgba": model.mat_rgba,
    "mat_emission": model.mat_emission,
    "mat_specular": model.mat_specular,
    "mat_shininess": model.mat_shininess,
    "mat_reflectance": model.mat_reflectance,
    # (nmat, mjNTEXROLE). All roles ship, not just albedo: MuJoCo's own renderer ignores
    # normal/roughness/metallic maps, so assets bind them purely for an external renderer --
    # which is what this bridge is. Without the albedo column alone the Mars terrain renders
    # as its material's flat white base colour.
    "mat_texid": model.mat_texid,
    "mat_texrepeat": model.mat_texrepeat,
    # texuniform changes primitive texture coordinates from normalized mesh UVs to physical local
    # coordinates. It must cross the bridge with texrepeat: workbench wood uses it to keep grain
    # density independent of its strongly non-uniform box dimensions.
    "mat_texuniform": model.mat_texuniform,
    # Lighting, shipped so Blender can reproduce the model's own setup instead of guessing at a
    # hardcoded sun. ``light_castshadow`` is deliberately NOT shipped: run.py zeroes it on the
    # whole model to work around the native viewer's lack of a per-geom shadow opt-out, and
    # exporting that workaround would strip every shadow from the photoreal render too. Blender
    # has the per-geom control (``geom_noshadow`` above), so it always shadows.
    "light_type": model.light_type,
    "light_diffuse": model.light_diffuse,
    "light_specular": model.light_specular,
    "light_cutoff": model.light_cutoff,
    "light_exponent": model.light_exponent,
    "light_active": model.light_active,
    "light_pos0": data.light_xpos.astype(np.float32),
    "light_dir0": data.light_xdir.astype(np.float32),
    # MuJoCo's headlight is a viewer-camera-attached light. The photoreal grade keeps only its
    # ambient term, scaled down, because reproducing the diffuse lobe flattens the frame; the
    # ``mujoco_look`` grade keeps both, because there the whole point is to agree with MuJoCo's
    # exposure. Both terms therefore ship and the Blender side picks. See ``_world_and_lights``.
    "headlight_ambient": np.asarray(model.vis.headlight.ambient, dtype=np.float32),
    "headlight_diffuse": np.asarray(model.vis.headlight.diffuse, dtype=np.float32),
    # Sky radiance, as set by ``visual_quality.apply_planetary_lighting``. MuJoCo has no
    # background-colour field, so that function parks it on ``vis.rgba.haze`` -- the atmosphere
    # colour, which is inert without a skybox and left at its white default by every scene here.
    # Blender reads a pure-white haze as "no sky authored" and falls back to the ambient-derived
    # background, so scenes that never call that function are untouched.
    "sky_rgb": np.asarray(model.vis.rgba.haze[:3], dtype=np.float32),
    "mujoco_look": np.bool_(mujoco_look),
    "render_size": np.asarray(size, dtype=np.int32),
    # Camera framing, in MuJoCo's own (lookat, azimuth, elevation, distance) parameterisation so
    # the photoreal window opens on the shot the native viewer would show for this ViewerConfig.
    # ``fovy`` ships too: distance alone does not determine framing.
    "cam_lookat0": _lookat(data, cfg, track_bid),
    "cam_azimuth": np.float32(cfg.azimuth),
    "cam_elevation": np.float32(cfg.elevation),
    "cam_distance": np.float32(cfg.distance),
    "cam_fovy": np.float32(cfg.fovy if cfg.fovy is not None else model.vis.global_.fovy),
  }

  for tid in range(model.ntex):
    adr, h, w = model.tex_adr[tid], model.tex_height[tid], model.tex_width[tid]
    nchan = model.tex_nchannel[tid]
    out[f"tex{tid}_data"] = model.tex_data[adr : adr + h * w * nchan].reshape(h, w, nchan)

  # Mesh and heightfield payloads are ragged, so they go in as one array per id rather than a
  # padded block. Only the ones actually referenced by a visible geom are worth shipping.
  for gid, gtype, dataid in zip(vis, model.geom_type[vis], model.geom_dataid[vis]):
    del gid
    if dataid < 0:
      continue
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
      va, vn = model.mesh_vertadr[dataid], model.mesh_vertnum[dataid]
      fa, fn = model.mesh_faceadr[dataid], model.mesh_facenum[dataid]
      out[f"mesh{dataid}_vert"] = model.mesh_vert[va : va + vn].astype(np.float32)
      out[f"mesh{dataid}_face"] = model.mesh_face[fa : fa + fn].astype(np.int32)
      # MuJoCo's compiler recomputes vertex normals with its own smoothing angle, and the bake
      # deliberately strips the OBJ's `vn` lines to rely on that (asset/srb_mars/README.md).
      # Letting Blender re-derive them instead rounds every rock edge off, so ship MuJoCo's.
      na = model.mesh_normaladr[dataid]
      out[f"mesh{dataid}_normal"] = model.mesh_normal[na : na + model.mesh_normalnum[dataid]].astype(np.float32)
      out[f"mesh{dataid}_facenormal"] = model.mesh_facenormal[fa : fa + fn].astype(np.int32)
      # UVs are indexed independently of vertices in MuJoCo (a seam vertex carries several),
      # so the per-face-corner index array has to ship alongside the coordinates.
      if model.mesh_texcoordadr[dataid] >= 0:
        ta, tn = model.mesh_texcoordadr[dataid], model.mesh_texcoordnum[dataid]
        out[f"mesh{dataid}_uv"] = model.mesh_texcoord[ta : ta + tn].astype(np.float32)
        out[f"mesh{dataid}_faceuv"] = model.mesh_facetexcoord[fa : fa + fn].astype(np.int32)
    elif gtype == mujoco.mjtGeom.mjGEOM_HFIELD:
      adr = model.hfield_adr[dataid]
      nrow, ncol = model.hfield_nrow[dataid], model.hfield_ncol[dataid]
      out[f"hfield{dataid}_data"] = (
        model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol).astype(np.float32)
      )
      out[f"hfield{dataid}_size"] = model.hfield_size[dataid].astype(np.float32)

  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **out)
  return path


def route_packet(points: np.ndarray | None) -> bytes:
  """Encode one sparse Blender-only planned-route update in world coordinates."""
  if points is None:
    return _ROUTE_HEADER.pack(_ROUTE_MAGIC, 0, 0)
  points = np.ascontiguousarray(points, dtype=np.float32)
  if points.ndim != 3 or points.shape[2] != 3 or not len(points) or not points.shape[1]:
    raise ValueError(f"route points need nonempty [paths, points, 3], got {points.shape}")
  if not np.isfinite(points).all():
    raise ValueError("route points must be finite")
  packet = _ROUTE_HEADER.pack(_ROUTE_MAGIC, *points.shape[:2]) + points.tobytes()
  if len(packet) > _MAX_PACKET:
    raise ValueError(f"route packet {len(packet)} bytes exceeds {_MAX_PACKET}-byte datagram limit")
  return packet


def pose_packet(model, data, dyn, cfg: ViewerConfig, track_bid: int, cam=None) -> np.ndarray:
  """One frame's wire payload: geom block, light tail, camera-pivot tail.

  ``dyn`` is the *moving* geom ids -- ``vis`` filtered by ``dynamic_geom_mask``, not ``vis``
  itself. Geoms welded to the world are already in the right place from the scene export and
  re-sending them every frame is the single largest avoidable cost on both ends of the wire.

  The FOV overlay geoms (group 4) are ordinary geoms on the camera gimbal link, so they ride the
  geom block and track the gimbal with no special handling. Lights and the pivot are appended as
  fixed-length tails rather than sent as separate datagrams: one packet cannot tear, so the frame
  Blender draws is always internally consistent.

  ``cam`` (an ``MjvCamera``, recording only) takes over the whole camera: the pivot becomes its
  eased ``lookat`` and a further ``(azimuth, elevation, distance)`` is appended, so a replay
  reproduces a chase cam that moves every frame. That is what makes the Blender video framed
  identically to the ``mujoco.Renderer`` one -- same cam object, same tick. The live viewer sends
  no orbit: there it belongs to the user's mouse, and overwriting it 60x a second is what made an
  earlier camera-locked version unnavigable.
  """
  parts = [
    np.hstack([data.geom_xpos[dyn], data.geom_xmat[dyn]]).ravel(),
    np.hstack([data.light_xpos, data.light_xdir]).ravel(),
    np.asarray(cam.lookat) if cam is not None else _lookat(data, cfg, track_bid),
  ]
  if cam is not None:
    parts.append(np.array([cam.azimuth, cam.elevation, cam.distance]))
  return np.concatenate(parts).astype(np.float32)


class BlenderStream:
  """Owns a live Blender process and the datagram channel feeding it one MuJoCo model's poses.

  Split out of ``BlenderViewer`` because the viewer assumes an mjlab env (GPU sim, ``env.cfg``,
  ``scene.entities``) while the cuRobo verify harnesses hold nothing but a compiled ``mj_model``
  and an ``mj_data`` they advance themselves. Both want the identical scene export, subprocess
  and packet format, so that part lives here and each caller keeps only its own state plumbing.

  Assumptions:
      * ``blender`` is on ``PATH`` and is 4.2+ (EEVEE Next).
      * A display is available; this is an interactive viewer by definition.
      * ``data`` is kinematically up to date when ``send`` is called (the caller owns stepping).
  """

  def __init__(
    self,
    model,
    data,
    cfg: ViewerConfig,
    track_bid: int,
    tag: str = "view",
    *,
    geomgroup=None,
    mujoco_look: bool = False,
  ):
    blender = shutil.which("blender")
    if blender is None:
      raise RuntimeError("`blender` not found on PATH; the photoreal viewer needs Blender 4.2+")
    self.model, self.cfg, self.track_bid = model, cfg, track_bid
    vis = visible_geom_ids(model, geomgroup)
    self.dyn = vis[dynamic_geom_mask(model, vis)]
    # Per-process paths so concurrent plays do not fight. The first version used a fixed UDP
    # port and a fixed scene file: a second run died with "Address already in use" and, worse,
    # silently overwrote the first run's scene. A Unix datagram socket needs no port
    # allocation and no free-port race -- the path is unique by construction.
    self._scene_path = _SCENE_DIR / f"scene_{tag}_{os.getpid()}.npz"
    self._sock_path = _SCENE_DIR / f"pose_{tag}_{os.getpid()}.sock"
    self._route_sock_path = _SCENE_DIR / f"route_{tag}_{os.getpid()}.sock"
    export_scene(
      model, data, cfg, track_bid, self._scene_path,
      geomgroup=geomgroup, mujoco_look=mujoco_look,
    )
    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    self._proc = subprocess.Popen(
      [blender, "--python", str(_BLENDER_SCRIPT), "--", str(self._scene_path), str(self._sock_path),
       str(self._route_sock_path)]
    )
    print(f"[INFO] Photoreal viewer: Blender pid {self._proc.pid}, streaming to {self._sock_path}")

  def send(self, data) -> None:
    packet = pose_packet(self.model, data, self.dyn, self.cfg, self.track_bid)
    try:
      self._sock.sendto(packet.tobytes(), str(self._sock_path))
    except OSError:
      pass  # Blender not listening yet, or already gone; is_running() handles the latter.

  def send_route(self, points: np.ndarray | None) -> None:
    """Send one planned-trajectory replacement; no work on normal pose ticks."""
    try:
      self._sock.sendto(route_packet(points), str(self._route_sock_path))
    except OSError:
      pass  # Blender not listening yet, or already gone.

  def is_running(self) -> bool:
    return self._proc.poll() is None

  def close(self) -> None:
    if self._proc.poll() is None:
      self._proc.terminate()
    self._sock.close()
    self._scene_path.unlink(missing_ok=True)
    self._sock_path.unlink(missing_ok=True)
    self._route_sock_path.unlink(missing_ok=True)


class BlenderRecorder:
  """Offscreen Blender video of a rollout: dump the pose stream now, render it headless after.

  Why not stream to a live Blender and grab its frames: EEVEE needs ~0.16 s a frame while the
  graded rollout produces one every ``control_dt``, so a live channel would drop most frames (a
  datagram socket) or throttle the physics being graded (a synchronous one). Neither is
  acceptable -- the video must be the graded run, frame for frame. Appending ~2 kB of poses per
  tick costs nothing, and the render then runs at whatever pace it likes once the verdict is in.

  Same rollout, same chase cam and same output path as the ``mujoco.Renderer`` recorder it
  parallels; only the renderer differs.

  ``cam`` is the caller's live ``MjvCamera``, read per frame -- so tracking is the chase cam's own
  eased ``lookat``, not a tracked body, and ``ViewerConfig`` here is WORLD (no tracking) purely to
  seed the still-render fallback pose in the scene export.

  ``opt`` is the caller's ``MjvOption`` and ``size`` its render resolution, both taken so the clip
  frames the same geoms at the same size as the ``mujoco.Renderer`` clip. ``mujoco_look`` defaults
  on for the same reason: two videos of one rollout that disagree on exposure cannot be compared.
  Turn it off for a clip with no MuJoCo counterpart (a figure/supplementary video), where the
  photoreal grade is the point.
  """

  def __init__(
    self, model, data, cam, out_path: str, fps: int, opt=None, size=_RENDER_SIZE,
    *, mujoco_look: bool = True,
  ):
    self.model = model
    self.cfg = ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD, lookat=tuple(cam.lookat),
      azimuth=cam.azimuth, elevation=cam.elevation, distance=cam.distance,
    )
    geomgroup = None if opt is None else opt.geomgroup
    vis = visible_geom_ids(model, geomgroup)
    self.dyn = vis[dynamic_geom_mask(model, vis)]
    self.out_path, self.fps = out_path, fps
    self._frames: list[np.ndarray] = []
    self._route_frames: list[int] = []
    self._routes: list[np.ndarray | None] = []
    # id(self), not just the pid: one rollout can feed several recorders (one per camera), and
    # they share a process -- the first close() would otherwise delete the scene the rest need.
    self._tag = f"{os.getpid()}_{id(self):x}"
    self._scene_path = _SCENE_DIR / f"scene_rec_{self._tag}.npz"
    export_scene(
      model, data, self.cfg, -1, self._scene_path,
      geomgroup=geomgroup, mujoco_look=mujoco_look, size=size,
    )

  def append(self, data, cam, route=_UNCHANGED) -> None:
    """Record one pose; route replacements carry static world paths only once."""
    self._frames.append(pose_packet(self.model, data, self.dyn, self.cfg, -1, cam))
    if route is not _UNCHANGED:
      self._route_frames.append(len(self._frames) - 1)
      self._routes.append(route)

  def start_render(self):
    """Spawn the headless Blender replay processes and return without waiting.

    Split out of ``close`` so a caller recording several cameras off one rollout can have all of
    their renders in flight at once. Every path this touches is keyed by ``self._tag``, so
    concurrent recorders share nothing but the GPU. Returns None when there is nothing to render,
    otherwise the process list for ``finish_render``.

    **Sharded across processes.** Most of a replay frame is Blender's own per-frame render cost,
    not this scene's: an empty scene costs 67 ms a frame at 720p and 202 ms at 1440p in the same
    background animation session, against 96 ms and 193 ms for a 215-object one. That floor is
    CPU-side and largely serial, so a second process hides it rather than contending for it --
    but only partly, because the shards still share one GPU. Measured on 240 frames at 720p,
    speedup by process count: 1.00 / 1.27 / 1.33 / 1.56 / 1.62 for 1 / 2 / 3 / 4 / 6, negative by
    8. ``_claim_shards`` therefore spends up to ``_MAX_REPLAY_PROCS`` on an idle recorder and
    stands down to one process each when several recorders are already in flight.
    """
    if not self._frames:
      return None
    frame_dir = _SCENE_DIR / f"frames_{self._tag}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    poses = _SCENE_DIR / f"poses_{self._tag}.npy"
    routes = _SCENE_DIR / f"routes_{self._tag}.npz"
    np.save(poses, np.stack(self._frames))
    route_data = np.empty(len(self._routes), dtype=object)
    route_data[:] = self._routes
    np.savez(routes, frames=np.asarray(self._route_frames, dtype=np.int32), routes=route_data)
    print(f"[record] rendering {len(self._frames)} frames in Blender (offscreen)...", flush=True)
    # Blender's bundled Python is /usr/bin/python3 and lacks numpy + scipy from our micromamba
    # env. Pass PYTHONPATH so blender_view.py can import them at --python launch. Locating the
    # env's site-packages relative to the running interpreter avoids a hard-coded /home/grl path
    # and survives renaming the env.
    import sys as _sys
    blender_env = dict(os.environ)
    py_path = next((p for p in _sys.path if p.endswith("site-packages")), None)
    if py_path:
      blender_env["PYTHONPATH"] = py_path + os.pathsep + blender_env.get("PYTHONPATH", "")
    n = _claim_shards(len(self._frames))
    bounds = np.linspace(0, len(self._frames), n + 1).astype(int)
    procs = [
      subprocess.Popen(
        [shutil.which("blender"), "-b", "--python", str(_BLENDER_SCRIPT), "--",
         str(self._scene_path), "-", "--replay", str(poses), str(frame_dir), str(routes),
         "--range", str(lo), str(hi - 1)],
        env=blender_env,
      )
      for lo, hi in zip(bounds[:-1], bounds[1:])
    ]
    _INFLIGHT.extend(procs)
    return procs

  def finish_render(self, procs) -> None:
    """Wait on every ``start_render`` shard, then mux their frames and clean up."""
    import imageio.v2 as imageio

    if procs is None:
      return
    # Exited shards are dropped from ``_INFLIGHT`` by the next ``_claim_shards``, not here: a
    # concurrent recorder claiming shards between this ``wait`` and a ``remove`` would already
    # have pruned the same object, and ``list.remove`` raises on a miss.
    for proc in procs:
      rc = proc.wait()
      if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)
    frame_dir = _SCENE_DIR / f"frames_{self._tag}"
    poses = _SCENE_DIR / f"poses_{self._tag}.npy"
    routes = _SCENE_DIR / f"routes_{self._tag}.npz"
    os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
    # Rate control is CRF rather than imageio's ``quality`` knob. The shipped walkthrough MP4s came
    # out at ~2.2 Mb/s for 2560x1440p50, low enough to macroblock the gripper/cube contact on
    # motion, which is the one thing these clips exist to show. CRF 14 is effectively transparent
    # for clean synthetic render output and, unlike a fixed quantiser, lets the bitrate track scene
    # complexity; ``preset slow`` spends encode time rather than bits for the same quality.
    # ``yuv420p`` is not optional -- x264's default for RGB input is yuv444p, which QuickTime,
    # PowerPoint and most browsers refuse to decode.
    with imageio.get_writer(self.out_path, fps=self.fps, codec="libx264", quality=None,
                            pixelformat="yuv420p",
                            output_params=["-crf", "14", "-preset", "slow"]) as w:
      # Whatever ``_replay`` wrote; the glob is format-agnostic so the intermediate codec stays a
      # Blender-side decision. Sorted by the frame *number*, not by name: Blender pads to four
      # digits, so a clip past 9999 frames (200 s at 50 fps) writes ``f10000`` -- which sorts
      # before ``f1000`` as text and would scramble the second half of the video silently.
      for frame in sorted(frame_dir.glob("f*"), key=lambda p: int(p.stem[1:])):
        w.append_data(imageio.imread(frame))
        frame.unlink()
    frame_dir.rmdir()
    poses.unlink(missing_ok=True)
    routes.unlink(missing_ok=True)
    self._scene_path.unlink(missing_ok=True)
    print(f"[record] wrote {self.out_path} ({len(self._frames)} frames @ {self.fps} FPS)", flush=True)

  def close(self) -> None:
    """Render every recorded frame in a headless Blender, then mux to ``out_path``."""
    self.finish_render(self.start_render())


class BlenderViewer(BaseViewer):
  """Drives a Blender GUI as the display for a policy rollout.

  Steps physics through ``BaseViewer``'s budget accumulator exactly like the native viewer, but
  instead of drawing, copies one environment's state onto a CPU ``MjData``, runs forward
  kinematics for world geom poses, and datagrams them to Blender.

  Assumptions:
      * A display is available; this is an interactive viewer by definition.
      * Single environment shown at a time (``env_idx``), matching the native viewer default.
  """

  def __init__(
    self,
    env,
    policy,
    frame_rate: float = 60.0,
    env_idx: int = 0,
    *,
    geomgroup=None,
    mujoco_look: bool = False,
    **kwargs,
  ):
    super().__init__(env, policy, frame_rate=min(frame_rate, _BLENDER_TIMER_HZ), **kwargs)
    self.env_idx = env_idx
    self._geomgroup, self._mujoco_look = geomgroup, mujoco_look
    self._stream: BlenderStream | None = None
    self._mjm = None
    self._mjd = None
    # Off by default: dynamic Blender path is scene-only (see [view] log line in
    # curobo_reach_verify), so the policy attribute may not even exist. Set MJ_ROUTE_OVERLAY=1
    # to send the planned cuRobo trajectory through the same sparse route datagram the
    # kinematic Blender loop already uses; only sends on identity change of viz_route.
    self._route_overlay = os.environ.get("MJ_ROUTE_OVERLAY") == "1"
    self._seen_route: object | None = None

  def setup(self) -> None:
    self._mjm = self.env.unwrapped.sim.mj_model
    self._mjd = mujoco.MjData(self._mjm)
    bid = track_body_id(self._mjm, self.cfg, self.env.unwrapped.scene.entities)
    self._pull_state()
    self._stream = BlenderStream(
      self._mjm, self._mjd, self.cfg, bid,
      geomgroup=self._geomgroup, mujoco_look=self._mujoco_look,
    )

  def _pull_state(self) -> None:
    """Copy one env's qpos off the GPU and map it to world geom poses.

    ``mj_kinematics``, not ``mj_forward``: everything the bridge consumes is ``geom_xpos`` and
    ``geom_xmat``, which kinematics alone fills (verified bit-identical to ``mj_forward`` over
    randomised qpos). ``mj_forward`` would additionally run collision detection against the
    Mars mesh terrain plus the constraint solve and inverse dynamics, all discarded -- 0.203 ms
    against 0.002 ms on ArgusMini20VelMarsMeshTerrain. That cost lands on the physics thread,
    where it is stolen straight from the sim-step budget.

    qvel is not copied for the same reason: kinematics does not read it, and skipping it halves
    the per-frame GPU syncs.

    ``mj_comPos``+``mj_camlight`` are the exception, added for the lights: kinematics places only
    ``mjCAMLIGHT_FIXED`` lights, and mjlab's terrain generator adds a spot in ``TRACKCOM`` mode
    that follows the robot. Left unevaluated it would stay at the origin and the robot would walk
    out of its own pool of light. Measured 0.0018 -> 0.0026 ms, against 0.191 ms for mj_forward.
    """
    self._mjd.qpos[:] = self.env.unwrapped.sim.data.qpos[self.env_idx].cpu().numpy()
    mujoco.mj_kinematics(self._mjm, self._mjd)
    mujoco.mj_comPos(self._mjm, self._mjd)  # subtree_com, which mj_camlight's TRACKCOM reads
    mujoco.mj_camlight(self._mjm, self._mjd)

  def sync_env_to_viewer(self) -> None:
    self._pull_state()
    self._stream.send(self._mjd)
    if self._route_overlay:
      self._maybe_send_route()

  def _maybe_send_route(self) -> None:
    """Send the planned cuRobo trajectory to Blender on identity change only.

    Same shape as ``curobo_reach_harness._run_reach_viewer_blender``: world-space origins of
    each reach's sampled waypoints, sent once when the policy's ``viz_route`` swaps in.
    Identity sentinel avoids per-tick datagram cost; clear on ``None``.

    ``viz_source``: viewers are often handed a closure that wraps the real policy (see
    ``curobo_reach_verify``'s ``viewer_policy``), which owns no route. That callsite tags the
    closure with the object that does.
    """
    route = getattr(getattr(self.policy, "viz_source", self.policy), "viz_route", None)
    if route is self._seen_route:
      return
    self._seen_route = route
    if route is None:
      self._stream.send_route(None)
      return
    from tasks.visual_manipulation.curobo_reach_harness import route_path_triads
    points = np.asarray(
      [p[0] for p in route_path_triads(self._mjm, route)], dtype=np.float32,
    ).reshape(-1, len(route.reaches), 3).transpose(1, 0, 2)
    self._stream.send_route(points)

  def sync_viewer_to_env(self) -> None:
    """No inbound channel: Blender is display-only, all control stays in the terminal."""

  def is_running(self) -> bool:
    return self._stream is not None and self._stream.is_running()

  def close(self) -> None:
    if self._stream is not None:
      self._stream.close()


class BlenderRecordViewer(BlenderViewer):
  """Record a play rollout as a photoreal Blender video instead of streaming it to a live one.

  Reuses ``BlenderViewer``'s state pull (one env's qpos -> world geom poses) and
  ``BlenderRecorder``'s dump-now/render-later split, so the only thing this adds is the loop.

  **Step-counted, not wall-clock.** ``BaseViewer.tick`` samples a frame off the real clock, which
  is right for a viewer and wrong for a recording: any tick the sim misses its realtime budget
  duplicates or drops a frame, so the clip would not be ``steps`` of sim time and the motion would
  stutter. Here one frame is captured every ``_spf`` env steps and the clip's fps is *derived*
  from that, so playback is real time by construction. ``run`` is overridden rather than
  ``tick``-driven for the same reason -- there is no display to pace against.

  ``target_fps`` is a request, not a promise: ``_spf`` is an integer number of control steps, so
  the achieved rate is the nearest ``1/(n*step_dt)``. At 50 Hz control, 30 -> 25 fps.

  Camera is the env's ``ViewerConfig`` orbit, re-aimed each frame at the tracked body -- the shot
  the native viewer would show, without a mouse to move it.
  """

  def __init__(self, env, policy, out_path: str, steps: int, target_fps: float = 30.0, **kwargs):
    super().__init__(env, policy, **kwargs)
    self._out_path, self._steps = out_path, steps
    step_dt = env.unwrapped.step_dt
    self._spf = max(1, round(1.0 / (target_fps * step_dt)))
    self._fps = round(1.0 / (self._spf * step_dt))
    self._cam = None
    self._rec: BlenderRecorder | None = None

  def setup(self) -> None:
    self._mjm = self.env.unwrapped.sim.mj_model
    self._mjd = mujoco.MjData(self._mjm)
    self._track_bid = track_body_id(self._mjm, self.cfg, self.env.unwrapped.scene.entities)
    self._pull_state()
    self._cam = mujoco.MjvCamera()
    self._cam.azimuth = self.cfg.azimuth
    self._cam.elevation = self.cfg.elevation
    self._cam.distance = self.cfg.distance
    self._cam.lookat[:] = _lookat(self._mjd, self.cfg, self._track_bid)
    self._rec = BlenderRecorder(
      self._mjm, self._mjd, self._cam, self._out_path, self._fps,
      size=(self.cfg.width, self.cfg.height), mujoco_look=self._mujoco_look,
    )
    print(
      f"[record] {self._steps} steps -> {self._steps // self._spf} frames @ {self._fps} FPS, "
      f"{self.cfg.width}x{self.cfg.height}"
    )

  def run(self, num_steps=None, catch_sigint=True) -> None:
    del num_steps, catch_sigint  # the clip length is `steps`, and the render must not be aborted
    self.setup()
    try:
      for i in range(self._steps):
        if not self._execute_step():
          break
        if i % self._spf == 0:
          self._pull_state()
          self._cam.lookat[:] = _lookat(self._mjd, self.cfg, self._track_bid)
          self._rec.append(self._mjd, self._cam)
    finally:
      self.close()

  def is_running(self) -> bool:
    return True

  def close(self) -> None:
    if self._rec is not None:
      self._rec.close()
      self._rec = None
