"""Project-local optimizations for delayed-builtin actuator processing.

Provides drop-in subclasses for mjlab's CircularBuffer and DelayBuffer
that eliminate GPU-CPU syncs in observation delay buffers — without modifying
the library.

Assumptions:
  - OptimizedCircularBuffer.reset() receives None, list, or Tensor (not slice).
    DelayBuffer.reset() converts slice → list before calling buffer.reset().
  - _current_length is NOT zeroed on reset. DelayBuffer zeroes _current_lags instead,
    so stale length values can't cause wrong lag reads before the next append.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mjlab.utils.buffers import CircularBuffer, DelayBuffer


# ---------------------------------------------------------------------------
# Optimized buffers
# ---------------------------------------------------------------------------


class OptimizedCircularBuffer(CircularBuffer):
  """CircularBuffer with Python-level backfill tracking to eliminate GPU-CPU syncs.

  Overrides append, __getitem__, and reset to avoid:
    - torch.any(is_first_push): GPU→CPU sync on every append
    - torch.all(valid == 0): GPU→CPU sync on every __getitem__ call
  Also caches current_length in-place (no tensor allocation per access).
  """

  def __init__(self, max_len: int, batch_size: int, device: str) -> None:
    super().__init__(max_len, batch_size, device)
    self._current_length = torch.zeros(batch_size, dtype=torch.long, device=device)
    # Python-level flags avoid GPU sync (torch.any) on every step.
    # _all_need_backfill: True after init or full reset, cleared on first append.
    # _pending_backfill_ids: specific env indices needing backfill after partial reset.
    self._all_need_backfill: bool = True
    self._pending_backfill_ids: torch.Tensor | None = None

  @property
  def current_length(self) -> torch.Tensor:
    return self._current_length

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    super().reset(batch_ids)  # zeros _num_pushes and buffer
    torch.minimum(self._num_pushes, self._max_len_tensor, out=self._current_length)
    if batch_ids is None:
      self._all_need_backfill = True
      self._pending_backfill_ids = None
    else:
      ids_tensor = (
        batch_ids
        if isinstance(batch_ids, torch.Tensor)
        else torch.as_tensor(list(batch_ids), dtype=torch.long, device=self._device)
      )
      self._pending_backfill_ids = (
        ids_tensor
        if self._pending_backfill_ids is None
        else torch.cat([self._pending_backfill_ids, ids_tensor])
      )

  def append(self, data: torch.Tensor) -> None:
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

    data = data.to(self._device)

    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self._max_len, *data.shape), dtype=data.dtype, device=self._device
      )

    self._pointer = (self._pointer + 1) % self._max_len

    # After warmup, both flags are False/None.
    if self._all_need_backfill:
      self._buffer[:] = data.unsqueeze(0)
      self._all_need_backfill = False
    else:
      self._buffer[self._pointer] = data
      if self._pending_backfill_ids is not None:
        self._buffer[:, self._pending_backfill_ids] = data[self._pending_backfill_ids]
        self._pending_backfill_ids = None

    self._num_pushes += 1
    torch.minimum(self._num_pushes, self._max_len_tensor, out=self._current_length)

  def backfill(self, data: torch.Tensor, batch_ids: torch.Tensor) -> None:
    """Backfill reset rows, keeping the Python-level tracking in sync.

    mjlab 1.6.0 added backfill() so a partial reset fills only the reset rows'
    history without advancing the global pointer. The parent writes
    _num_pushes[batch_ids] = 1 directly, so this override must (a) refresh the
    _current_length cache the parent doesn't know about, and (b) drop the
    deferred backfill our reset() queued — the parent has now done that work
    eagerly, and letting append() redo it would overwrite the reset frame (and
    the one real step of history after it) with the newest frame.

    Assumes the caller pairs reset(env_ids) with backfill(_, env_ids), which is
    how ObservationManager._reset_buffers drives it.
    """
    super().backfill(data, batch_ids)
    torch.minimum(self._num_pushes, self._max_len_tensor, out=self._current_length)
    self._pending_backfill_ids = None

  def __getitem__(self, key: torch.Tensor | int) -> torch.Tensor:
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")

    if isinstance(key, int):
      key = torch.full((self._batch_size,), key, dtype=torch.long, device=self._device)
    else:
      if key.ndim == 0:
        key = key.expand(self._batch_size)
      key = key.to(device=self._device, dtype=torch.long)

    if key.numel() != self._batch_size:
      raise ValueError(f"Expected {self._batch_size} lags, got {key.numel()}")

    # Clamp to the oldest RETAINED frame (mjlab 1.6.0 fix, ported): without the
    # max_len bound, a lag past the buffer length wraps to a newer frame once
    # num_pushes exceeds max_len.
    pushes = self._num_pushes.clamp_min(1)
    max_lag = torch.minimum(pushes, self._max_len_tensor) - 1
    valid = torch.minimum(key, max_lag).clamp_min(0)
    # No torch.all(valid == 0) GPU-CPU sync — advanced 2D indexing runs asynchronously
    # on GPU and guarantees a standalone COPY tensor (no view aliasing bugs).
    idx = torch.remainder(self._pointer - valid, self._max_len)
    return self._buffer[idx, self._all_indices]


class OptimizedDelayBuffer(DelayBuffer):
  """DelayBuffer backed by OptimizedCircularBuffer, with fast path for update_period=0.

  Replaces the internal CircularBuffer after parent __init__, and overrides _update_lags
  to skip the should_update mask when all envs update every step.
  """

  def __init__(
    self,
    min_lag: int = 0,
    max_lag: int = 3,
    batch_size: int = 1,
    device: str = "cpu",
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
    generator: torch.Generator | None = None,
  ) -> None:
    super().__init__(min_lag, max_lag, batch_size, device, per_env, hold_prob,
                     update_period, per_env_phase, generator)
    # Replace the CircularBuffer created by parent with the sync-free version.
    buffer_size = max_lag + 1 if max_lag > 0 else 1
    self._buffer = OptimizedCircularBuffer(
      max_len=buffer_size, batch_size=batch_size, device=device
    )

  def _update_lags(self) -> None:
    if self.update_period > 0:
      phase_adjusted_count = (self._step_count + self._phase_offsets) % self.update_period
      should_update = phase_adjusted_count == 0
      new_lags = self._sample_lags(should_update)
      self._current_lags = torch.where(should_update, new_lags, self._current_lags)
      self._step_count += 1
    else:
      # Fast path: skip mask allocation and step_count bookkeeping.
      self._current_lags = self._sample_lags_all()

  def _sample_lags_all(self) -> torch.Tensor:
    """Sample new lags for all envs — fast path for update_period=0."""
    if self.per_env:
      candidate_lags = torch.randint(
        self.min_lag, self.max_lag + 1, (self.batch_size,),
        dtype=torch.long, device=self.device, generator=self.generator,
      )
    else:
      candidate_lags = torch.randint(
        self.min_lag, self.max_lag + 1, (1,),
        dtype=torch.long, device=self.device, generator=self.generator,
      ).expand(self.batch_size).clone()

    if self.hold_prob > 0.0:
      should_sample = (
        torch.rand(self.batch_size, dtype=torch.float32, device=self.device,
                   generator=self.generator)
        >= self.hold_prob
      )
      return torch.where(should_sample, candidate_lags, self._current_lags)

    return candidate_lags


# ---------------------------------------------------------------------------
# Observation manager patch
# ---------------------------------------------------------------------------

def _patch_observation_manager() -> None:
  """Replace CircularBuffer/DelayBuffer in the observation manager module.

  The observation manager creates CircularBuffer (for history) and DelayBuffer
  (for delayed observation terms like joint_pos/joint_vel) in its __init__.
  The original implementations have torch.any()/torch.all() GPU→CPU syncs
  that block Python from dispatching the next step until all GPU kernels finish.

  We patch the module-level names BEFORE any env is created (this file is
  imported via mdp_helpers.py which is loaded before ManagerBasedRlEnv is
  instantiated). Patching module-level names is safe here because:
    - The observation manager binds these names at import time; __init__ reads them
      from the module namespace, not from a closure, so our replacement takes effect.
    - We do NOT modify any mjlab source files.
  """
  import mjlab.managers.observation_manager as _obs_mgr
  _obs_mgr.CircularBuffer = OptimizedCircularBuffer
  _obs_mgr.DelayBuffer = OptimizedDelayBuffer


_patch_observation_manager()
