"""Scripted reference-path tracking for measurable play recordings.

Play resamples the velocity command at random every 2-8 s, so a recorded rollout is a random
walk: it looks alive but carries no number, and a figure captioned "the policy walks around"
proves nothing. Here the command follows an analytic closed path instead, which turns the same
clip into a measurement -- the reference curve is drawn on the terrain next to the path the
robot actually traced, and the gap between them is reported as a cross-track error.

**World frame throughout**, because this policy is world-frame: ``commands_xy`` observes
``vel_command_w`` directly and the velocity observations are ``base_*_vel_world``, so writing
``vel_command_w`` writes exactly the signal the policy was trained to follow. ``rel_world_envs=0``
is left alone -- it only disables the body-frame *copy* in ``_update_command``, which nothing in
this task's observation or reward graph reads.

**Closed loop, not open loop.** Integrating an open-loop command over 30 s of rough Mars terrain
drifts without bound, so the traced path would diverge from the reference for reasons that say
nothing about tracking quality and the error would grow with clip length rather than converge to
a property of the policy. The command here is ``v*tangent + kp*(p_ref - p)``: a lag pulls the
robot back onto the curve, so cross-track error stays a bounded, clip-length-independent number.
The reference itself advances on a clock (constant arc-length rate), not on the robot's progress,
so the metric still penalises a policy that cannot keep up -- a pure-pursuit parameterisation
would let a slow policy score perfectly by simply arriving late.

``max_speed`` clamps the command to the trained range; the headroom between ``speed`` and
``max_speed`` is what the feedback term has to catch up with, so keep ``speed`` below it.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import mujoco

from photoreal.bridge import BlenderRecorder, BlenderRecordViewer, _lookat

PATHS = ("figure8", "circle", "square", "line")

_ROUTE_POINTS = 256  # per drawn curve; a route packet is one rectangular [paths, points, 3]
_RAY_TOP = 30.0  # m above the sampled point to start the downward ground ray, clear of relief
_STRIDE_S = 0.5  # averaging window for the gait-cycle-mean velocity reported alongside the raw
_ROUTE_LIFT = 0.02  # m above the ground hit, so the tube is not z-fighting the regolith
# Chassis-ground contact ends a training episode but should not cut a recording short; set
# MJ_TRAJ_END_ON_CONTACT=1 to record the failure instead of walking through it.
_ALLOW_BASE_CONTACT = os.environ.get("MJ_TRAJ_END_ON_CONTACT") != "1"


def _reference_xy(name: str, size: float, n: int = 2001) -> np.ndarray:
  """Closed reference polyline in local XY, centred on the origin.

  ``size`` is the bounding width in metres for every shape, so one flag rescales all of them.
  Dense (n=2001) because everything downstream -- constant-speed traversal, tangents, nearest-
  point error -- is computed on the polyline rather than on the closed form, which keeps one
  code path for four shapes.
  """
  u = np.linspace(0.0, 2.0 * np.pi, n)
  r = 0.5 * size
  if name == "figure8":  # lemniscate of Gerono: self-crossing, curvature of both signs
    return np.stack([r * np.cos(u), r * np.sin(u) * np.cos(u)], axis=1)
  if name == "circle":
    return np.stack([r * np.cos(u), r * np.sin(u)], axis=1)
  if name == "square":
    # Superellipse, not a true square: rounded corners keep the reference velocity finite, so
    # the measured error stays a property of the policy and not of a discontinuity we authored.
    p = 2.0 / 6.0
    return np.stack([
      r * np.sign(np.cos(u)) * np.abs(np.cos(u)) ** p,
      r * np.sign(np.sin(u)) * np.abs(np.sin(u)) ** p,
    ], axis=1)
  if name == "line":  # out and back; the reversal at each end is a deliberate command step
    return np.stack([r * np.cos(u), np.zeros_like(u)], axis=1)
  raise ValueError(f"unknown path {name!r}; expected one of {PATHS}")


class ReferencePath:
  """Arc-length parameterised world-frame path, anchored at the robot's start position."""

  def __init__(self, name: str, size: float, start_xy: np.ndarray):
    raw = _reference_xy(name, size)
    self.points = raw - raw[0] + start_xy  # start on the curve: no transient at t=0 to explain
    seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
    self._s = np.concatenate([[0.0], np.cumsum(seg)])
    self.length = float(self._s[-1])

  def at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent at arc length ``s``, wrapping around the closed path."""
    p = self._interp(s)
    ds = 1e-3 * self.length
    d = self._interp(s + ds) - p
    n = np.linalg.norm(d)
    return p, (d / n if n > 0 else np.zeros(2))

  def _interp(self, s: float) -> np.ndarray:
    s = s % self.length
    return np.array([np.interp(s, self._s, self.points[:, i]) for i in (0, 1)])

  def project(self, xy: np.ndarray, s_prev: float, back: float = 0.3, ahead: float = 1.5) -> float:
    """Arc length of the nearest path vertex, searched only in a window around ``s_prev``.

    A global argmin is WRONG for a self-intersecting path used as a controller reference: at the
    figure-eight crossing the two branches are metres apart in arc length but centimetres apart in
    space, so the robot snaps onto the opposite branch and follows a short-circuited loop that
    never reaches the far lobe. Cross-track error stays small the whole time -- it is still near
    *a* part of the path -- so the metric does not reveal it; only the plotted trace does.

    Restricting the search to ``[s_prev - back, s_prev + ahead]`` keeps progress monotone through
    the crossing. ``ahead`` must exceed the lookahead so the reference can advance, and ``back``
    stays small but non-zero so a lateral disturbance does not force fake forward progress.

    ``cross_track`` deliberately keeps its global search: as a *metric* the nearest point on the
    whole path is the right answer, and it carries no state to corrupt.
    """
    lo, hi = s_prev - back, s_prev + ahead
    s = np.arange(lo, hi, self.length / len(self.points))
    pts = np.stack([self._interp(v) for v in s])
    return float(s[int(np.argmin(np.linalg.norm(pts - xy, axis=1)))] % self.length)

  def curvature(self, s: float) -> float:
    """Menger curvature at arc length ``s``, from three points one percent of a lap apart.

    Used to slow the reference in the lobes. A finite-difference second derivative would need a
    step small enough to be dominated by the 2001-vertex discretisation; the circumscribed-circle
    form is stable at a step this large.
    """
    h = 0.01 * self.length
    a, b, c = self._interp(s - h), self._interp(s), self._interp(s + h)
    ab, bc, ca = (np.linalg.norm(x - y) for x, y in ((a, b), (b, c), (c, a)))
    area = abs(np.cross(b - a, c - a)) / 2.0
    denom = ab * bc * ca
    return 0.0 if denom < 1e-9 else 4.0 * area / denom

  def cross_track(self, xy: np.ndarray) -> np.ndarray:
    """Distance from each row of ``xy`` to the nearest point on the polyline.

    Brute force against all 2001 vertices: 1500 steps x 2001 vertices is 3M distances, which is
    milliseconds, and it needs no assumption about which segment is nearest (the figure-8 crosses
    itself, so a monotone-progress search would be wrong there).
    """
    return np.linalg.norm(xy[:, None, :] - self.points[None, :, :], axis=2).min(axis=1)


def _ground_z(model, data, x: float, y: float) -> float:
  """World z of the terrain under (x, y), by ray cast straight down.

  The robot is in the scene and the ray does not know that, so a hit on a non-world body is
  skipped by restarting the ray just past it -- three tries covers standing under the robot,
  which is exactly where the traced path is sampled every frame.
  """
  pnt = np.array([x, y, _RAY_TOP + float(data.qpos[2])], dtype=np.float64)
  vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  gid = np.zeros(1, dtype=np.int32)
  for _ in range(3):
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, gid)
    if dist < 0:
      break
    hit_z = pnt[2] - dist
    if gid[0] >= 0 and model.geom_bodyid[gid[0]] == 0:  # world body == terrain
      return hit_z
    pnt[2] = hit_z - 0.05
  return 0.0


class TrajRecordViewer(BlenderRecordViewer):
  """Record a path-tracking rollout from two cameras at once.

  One rollout, two ``BlenderRecorder``s: the pose packets differ only in the camera they carry,
  so a second camera costs a second Blender render pass and nothing in the physics. The wide
  camera is static and frames the whole path -- that is the shot the cross-track number refers
  to -- while the chase camera is the tracked orbit, for the gait detail the wide shot loses.

  Writes ``<stem>_track.npz`` (raw log) and ``<stem>_track.png`` (reference vs traced path,
  error over time, commanded vs measured velocity) next to the videos.
  """

  def __init__(
    self, env, policy, out_path: str, steps: int,
    *, path: str = "figure8", size: float = 4.0, speed: float = 0.5,
    max_speed: float = 0.8, lookahead: float = 0.5, **kwargs,
  ):
    # A 2 mm tube (the cuRobo default) is under a pixel wide from the wide camera; the headless
    # render inherits this env var through subprocess.run.
    os.environ.setdefault("MJ_ROUTE_RADIUS", "0.02")
    stem = out_path[:-4] if out_path.endswith(".mp4") else out_path
    # The output stem is derived from the task, so two runs of the same task (different seeds)
    # race on one _track.npz. MJ_TRAJ_TAG makes them distinct, which is what lets a whole seed
    # sweep run concurrently instead of one seed-round at a time.
    stem += os.environ.get("MJ_TRAJ_TAG", "")
    super().__init__(env, policy, f"{stem}_chase.mp4", steps, **kwargs)
    self._stem = stem
    self._path_name, self._size, self._speed = path, size, speed
    self._max_speed = max_speed
    # Lookahead sets the corner-cutting/oscillation trade: too short and the command chatters on
    # the gait's own lateral sway, too long and the lobes get cut. 0.5 m is ~0.8 s of travel at
    # 0.6 m/s and ~1.5 robot diameters.
    self._lookahead = lookahead
    self._s = 0.0  # arc length of the robot's projection, carried between steps
    self._progress = 0.0  # unwrapped arc length covered; ends the clip at exactly one lap
    self._ref: ReferencePath | None = None
    self._wide: BlenderRecorder | None = None
    self._log: list[np.ndarray] = []
    self._term_name: str | None = None  # which termination ended the clip, None if it ran to length
    self._term_time = float("nan")

  def setup(self) -> None:
    super().setup()  # chase cam + chase recorder, tracked exactly as the native viewer tracks
    robot = self.env.unwrapped.scene["robot"]
    start = robot.data.root_link_pos_w[self.env_idx, :2].cpu().numpy().astype(np.float64)
    self._ref = ReferencePath(self._path_name, self._size, start)

    term = self.env.unwrapped.command_manager._terms["twist"]
    term.resample = lambda env_ids: None  # the path owns the command; no random resampling
    term.is_standing_env[:] = False  # rel_standing_envs would zero the command mid-clip
    self._term = term

    # Let the chassis touch the ground without ending the clip. `base_contact` fires at 1 N of
    # housing-terrain force, which on a rocky surface is a graze against a rock the robot walks
    # out of; training needs that as a failure signal, a recording does not, and the reset it
    # triggers is what put a teleport in the middle of every Mars trace. Replacing the term's func
    # (rather than deleting the term) keeps the manager's buffers and logging shape intact.
    tm = self.env.unwrapped.termination_manager
    if _ALLOW_BASE_CONTACT and "base_contact" in tm.active_terms:
      cfg = tm.get_term_cfg("base_contact")
      cfg.func = lambda env, **_: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      cfg.params = {}

    centre = self._ref.points.mean(axis=0)
    self._wide_cam = mujoco.MjvCamera()
    self._wide_cam.azimuth = self.cfg.azimuth
    self._wide_cam.elevation = -40.0  # high enough to read the shape, low enough to keep relief
    # 1.6 framed the path itself but left no margin: at -40 degrees the near half of the path is
    # foreshortened downward, so the robot -- which starts and finishes on the path's near edge --
    # was clipped by the bottom of the frame in the closing frame of every clip. 1.9 buys that
    # margin at a modest cost in path size.
    self._wide_cam.distance = 1.9 * self._size
    self._wide_cam.lookat[:] = (
      centre[0], centre[1], _ground_z(self._mjm, self._mjd, centre[0], centre[1]),
    )
    self._wide = BlenderRecorder(
      self._mjm, self._mjd, self._wide_cam, f"{self._stem}_wide.mp4", self._fps,
      size=(self.cfg.width, self.cfg.height), mujoco_look=self._mujoco_look,
    )

    idx = np.linspace(0, len(self._ref.points) - 1, _ROUTE_POINTS).astype(int)
    ref_xy = self._ref.points[idx]
    self._ref_curve = np.stack([
      ref_xy[:, 0], ref_xy[:, 1],
      [_ground_z(self._mjm, self._mjd, x, y) + _ROUTE_LIFT for x, y in ref_xy],
    ], axis=1).astype(np.float32)
    self._trace: list[np.ndarray] = []
    print(
      f"[track] {self._path_name} size={self._size} m, {self._ref.length:.2f} m per lap at "
      f"{self._speed} m/s -> {self._steps * self.env.unwrapped.step_dt / (self._ref.length / self._speed):.2f} laps"
    )

  def _command(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-pursuit velocity command, and the reference point the error is measured against.

    Three laws were measured on the same 6 cells x 3 seeds; this one won and the losers are worth
    recording, because two of them look like obvious improvements.

    1. Clock-driven reference (``s = speed * elapsed``). Its dominant error is *lag*, not leaving
       the path: the robot stays on the curve but falls behind, and once behind, the command points
       at a receding target instead of along the path, which cuts the lobe corners. Its
       reference-lag RMS of 0.43-1.24 m is NOT a cross-track number and must never be quoted as
       one -- the same clip measured 0.106 m of true cross-track.
    2. Tangent feedforward plus cross-track feedback, ``speed * tangent + gain * (p_ref - xy)``.
       Principled -- no chord to cut, correction along the shortest way back -- and worse in
       practice. Putting the correction in the command's magnitude collapses forward progress once
       ``gain * |error|`` exceeds the speed: mean lap time went 31 s to 74 s over the gain sweep.
       Normalising to steer rather than add fixed the progress loss, but a 142-run gain sweep
       (0.5 to 32 /m, 3 seeds) bottomed out at 0.195 m mean and 69 s laps against pure pursuit's
       0.191 m at 40 s. Below gain 4 it is far worse; the whole curve is a wash at best.
    3. Pure pursuit, below. Projecting removes the lag term by construction: the reference sits a
       fixed ``lookahead`` ahead of wherever the robot actually is, so the only error left is
       lateral, and the lookahead gives a correction that softens with distance instead of
       fighting the gait's own lateral sway.

    Speed is scaled down where the path bends, by ``1 / (1 + lookahead * curvature)``. That factor
    is the ratio between the chord the robot would cut and the arc it should follow, so it slows
    exactly as much as the corner would otherwise cut. This is why the lobe apex -- the tightest
    curvature on a lemniscate and where every trace bulged outside the reference -- gets the
    biggest reduction.

    Direction comes from the lookahead point rather than the path tangent: this robot crab-walks
    with no yaw command, so it can accelerate straight at a goal point with no turning transient,
    which is the case pure pursuit is built for.
    """
    s = self._ref.project(xy, self._s)
    # Accumulate UNWRAPPED progress. Under pure pursuit the reference is carried by the robot, so
    # a lap takes path_length / ACHIEVED speed -- and achieved speed is ~0.45 m/s against a 0.6
    # command, which is why a clip sized off the command truncated every trace, each by a
    # different amount. `_progress` is what ends the clip instead; see run().
    # Unwrap BOTH ways. The search window reaches behind the current position, and its negative
    # part wraps to just under `length`, so step one can report s ~= 14.9 against s_prev = 0 --
    # a whole lap of fake progress before the robot has moved.
    ds = s - self._s
    if ds < -0.5 * self._ref.length:
      ds += self._ref.length
    elif ds > 0.5 * self._ref.length:
      ds -= self._ref.length
    self._progress += ds
    self._s = s  # monotone progress; the window search needs last step's arc length
    p_ref, _ = self._ref.at(s)  # nearest point on the path; what cross-track error is measured to
    goal, _ = self._ref.at(s + self._lookahead)
    to_goal = goal - xy
    n = np.linalg.norm(to_goal)
    if n < 1e-9:
      return p_ref, np.zeros(2)
    # Curvature slowdown, deliberately NOT floored at the measured unity-gain speed. Below
    # ~0.55 m/s these policies overshoot their command (achieved/commanded 1.39-1.46 at a
    # commanded 0.3, 1.5-1.8 at 0.15) because training samples (vx, vy) uniformly on a SQUARE and
    # so almost never visits low magnitudes. Flooring the command at 0.55 to stay out of that
    # band was TESTED AND REJECTED (18 runs, 2026-08-12): cross-track RMS went 0.273 -> 0.547 m
    # on dof12 Mars, 0.130 -> 0.210 on dof32 Mars, worse in five of six cells. Carrying unity
    # speed through the apex cuts the corner by more than the command-following error it removes.
    speed = self._speed / (1.0 + self._lookahead * self._ref.curvature(s))
    cmd = speed * (to_goal / n)
    norm = np.linalg.norm(cmd)
    if norm > self._max_speed:
      cmd *= self._max_speed / norm
    return p_ref, cmd

  def _fired_termination(self) -> str | None:
    """Name of the termination that just ended this env's episode, or None if it is still alive.

    Read after ``env.step``: the manager holds the flags for the step that just resolved, and the
    reset that follows them is what would corrupt the recording.
    """
    tm = self.env.unwrapped.termination_manager
    for name in tm.active_terms:
      if bool(tm.get_term(name)[self.env_idx]):
        return name
    return None

  def _route(self) -> np.ndarray:
    """Traced and reference path as one [2, N, 3] packet, both resampled to a common N.

    Traced first because the renderer colours curves by index and its first colour (cyan) is the
    one that survives on orange regolith; the reference is a smooth curve and reads in the second.
    """
    trace = np.asarray(self._trace, dtype=np.float32)
    idx = np.linspace(0, len(trace) - 1, _ROUTE_POINTS).astype(int)
    return np.stack([trace[idx], self._ref_curve])

  def run(self, num_steps=None, catch_sigint=True) -> None:
    del num_steps, catch_sigint  # clip length is `steps`; a render must not be aborted midway
    self.setup()
    robot = self.env.unwrapped.scene["robot"]
    dt = self.env.unwrapped.step_dt
    try:
      for i in range(self._steps):
        xy = robot.data.root_link_pos_w[self.env_idx, :2].cpu().numpy().astype(np.float64)
        p_ref, cmd = self._command(xy)
        # Written before the step: the observation the policy acts on is built inside env.step.
        self._term.vel_command_w[:, 0] = float(cmd[0])
        self._term.vel_command_w[:, 1] = float(cmd[1])
        self._term.vel_command_w[:, 2] = 0.0

        if not self._execute_step():
          break

        # `base_contact` is disabled for the clip (see setup), so nothing short of a timeout should
        # fire; if one does, stop rather than record the teleport a reset would draw across the
        # plot as a straight line and cut into the video mid-stride.
        fired = self._fired_termination()
        if fired is not None:
          self._term_name, self._term_time = fired, i * dt
          print(f"[track] episode ended at t={i * dt:.2f} s: {fired}", flush=True)
          break

        # One lap exactly, for every robot. `steps` is only a safety cap now: a slow robot needs
        # more wall time for the same arc length, and cutting it at a fixed step count truncated
        # each trace by a different amount, which is not a comparison.
        if self._progress >= self._ref.length:
          self._term_time = i * dt
          print(f"[track] lap complete at t={i * dt:.2f} s", flush=True)
          break

        vel = robot.data.root_link_lin_vel_w[self.env_idx, :2].cpu().numpy()
        self._log.append(np.concatenate([[i * dt], p_ref, xy, cmd, vel]))
        if i % self._spf == 0:
          self._pull_state()
          self._cam.lookat[:] = _lookat(self._mjd, self.cfg, self._track_bid)
          self._trace.append([
            xy[0], xy[1], _ground_z(self._mjm, self._mjd, xy[0], xy[1]) + _ROUTE_LIFT,
          ])
          # Route tubes go to the wide camera only. Their radius is set for a ~5 m shot; from the
          # chase camera's ~1 m the same tube is wider than the robot and hides what that clip
          # exists to show.
          self._rec.append(self._mjd, self._cam)
          self._wide.append(self._mjd, self._wide_cam, route=self._route())
    finally:
      self.close()

  def close(self) -> None:
    if self._log:
      self._report()
      self._log = []
    if os.environ.get("MJ_TRAJ_NORENDER") == "1":
      # Screening a spawn (does the robot complete the lap, or stall against a rock?) needs the
      # metrics and none of the two Blender passes those metrics would otherwise pay for.
      self._rec = self._wide = None
      return
    # Both renders in flight together. They are separate Blender processes over disjoint temp
    # paths, so the only thing they contend for is the GPU, and neither saturates it alone --
    # serially the wide pass sat idle for the whole chase pass and vice versa.
    chase_proc = self._rec.start_render() if self._rec is not None else None
    wide_proc = self._wide.start_render() if self._wide is not None else None
    if self._rec is not None:
      self._rec.finish_render(chase_proc)
      self._rec = None  # matches the parent's close(); this method replaces it, not extends it
    if self._wide is not None:
      self._wide.finish_render(wide_proc)
      self._wide = None

  def _report(self) -> None:
    """Save the raw log and a three-panel figure, and print the headline numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = np.asarray(self._log)
    t, p_ref, xy, cmd, vel = log[:, 0], log[:, 1:3], log[:, 3:5], log[:, 5:7], log[:, 7:9]
    err = self._ref.cross_track(xy)
    lag = np.linalg.norm(xy - p_ref, axis=1)  # includes along-path lag, unlike cross_track
    vel_err = np.linalg.norm(cmd - vel, axis=1)
    # A legged base oscillates with the stride, so the instantaneous world velocity swings far
    # wider than any command ever asks for and its RMSE measures the gait, not the tracking.
    # Averaging over one stride (~0.5 s) leaves the part of the error the command can explain;
    # both numbers are reported, because the raw one is what a controller downstream would see.
    win = max(1, int(round(_STRIDE_S / (t[1] - t[0])))) if len(t) > 1 else 1
    kernel = np.ones(win) / win
    vel_s = np.stack([np.convolve(vel[:, k], kernel, mode="same") for k in (0, 1)], axis=1)
    cmd_s = np.stack([np.convolve(cmd[:, k], kernel, mode="same") for k in (0, 1)], axis=1)
    vel_err_s = np.linalg.norm(cmd_s - vel_s, axis=1)
    np.savez(
      f"{self._stem}_track.npz", t=t, p_ref=p_ref, xy=xy, cmd=cmd, vel=vel,
      reference=self._ref.points, path=self._path_name, size=self._size, speed=self._speed,
      termination=self._term_name or "", term_time=self._term_time,
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(self._ref.points[:, 0], self._ref.points[:, 1], "k--", lw=1, label="reference")
    ax[0].plot(xy[:, 0], xy[:, 1], lw=1.6, label="tracked")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].legend(); ax[0].set_title(f"{self._path_name} @ {self._speed} m/s")
    ax[1].plot(t, err, lw=1)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("cross-track error [m]")
    ax[1].set_title(f"RMS {np.sqrt((err**2).mean()):.3f} m, max {err.max():.3f} m")
    for k, name in enumerate(("vx", "vy")):
      colour = f"C{k}"
      ax[2].plot(t, cmd[:, k], "--", lw=1.2, color=colour, label=f"cmd {name}")
      ax[2].plot(t, vel[:, k], lw=0.6, color=colour, alpha=0.3)
      ax[2].plot(t, vel_s[:, k], lw=1.4, color=colour, label=name)
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("world velocity [m/s]"); ax[2].legend(ncol=2)
    ax[2].set_title(
      f"vel RMSE {np.sqrt((vel_err_s**2).mean()):.3f} m/s stride-mean, "
      f"{np.sqrt((vel_err**2).mean()):.3f} raw"
    )
    fig.tight_layout()
    fig.savefig(f"{self._stem}_track.png", dpi=160)
    plt.close(fig)

    print(
      f"[track] cross-track RMS {np.sqrt((err**2).mean()):.3f} m, max {err.max():.3f} m | "
      f"reference lag RMS {np.sqrt((lag**2).mean()):.3f} m | "
      f"world-vel RMSE {np.sqrt((vel_err_s**2).mean()):.3f} m/s stride-mean "
      f"({np.sqrt((vel_err**2).mean()):.3f} raw) | "
      f"mean speed {np.linalg.norm(vel, axis=1).mean():.3f} m/s",
      flush=True,
    )
    print(f"[track] wrote {self._stem}_track.npz and {self._stem}_track.png", flush=True)
