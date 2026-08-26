"""PPO-side compact training logger (rsl_rl Logger subclass).

The shared template engine (_TemplateLogMixin) and the rsl_rl-free FlashSACCompactLogger now live in
flash_sac/compact_logger.py so the flash_sac package carries no rsl_rl dependency. CompactLogger stays
here because it subclasses rsl_rl.utils.logger.Logger; it imports the engine from flash_sac.
"""

import statistics
import time
import torch
from rsl_rl.utils.logger import Logger

from flash_sac.logger import _TemplateLogMixin


class CompactLogger(Logger, _TemplateLogMixin):
    """Compact rsl_rl Logger with grouped, fixed-width aligned output.

    On the first call (or when the extras key set changes), builds a rendering
    template that bakes key names and format strings into a list of instructions.
    Subsequent iterations just substitute values into pre-built format strings,
    avoiding per-iteration sorting, grouping, and string construction.
    """

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 90,
        pad: int = 40,
    ) -> None:
        """Log with compact console output (tensorboard writes unchanged)."""
        if not self.writer:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        # ── Tensorboard writes ──────────────────────────────────────────────────
        extras_dict: dict[str, float] = {}
        if self.ep_extras:
            for key in self.ep_extras[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                extras_dict[key] = float(value)
                if "/" in key:
                    self.writer.add_scalar(key, value, it)
                else:
                    self.writer.add_scalar("Episode/" + key, value, it)

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

        self.writer.add_scalar("Policy/mean_noise_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(self.rewbuffer) > 0:
            if self.cfg["algorithm"]["rnd_cfg"]:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)

        # ── Console output ──────────────────────────────────────────────────────
        # Reactive template check: O(1) fast path when key set is stable
        if (len(extras_dict) != getattr(self, "_n_keys", -1)
                or frozenset(extras_dict) != getattr(self, "_template_keys", None)):
            self._rebuild_template(extras_dict, width, print_minimal)

        # Dynamic header (iteration number changes)
        loss_str = "  ".join(f"{k.split()[0]} {v:7.4f}" for k, v in loss_dict.items())
        mean_rew = f"{statistics.mean(self.rewbuffer):7.2f}" if self.rewbuffer else "    N/A"
        mean_len = f"{statistics.mean(self.lenbuffer):6.2f}" if self.lenbuffer else "   N/A"
        task_label = getattr(self, "task_label", None)
        iter_label = (
            f" {task_label}  iter {it}/{total_it} "
            if task_label
            else f" iter {it}/{total_it} "
        )
        left = (width - len(iter_label)) // 2
        right = width - len(iter_label) - left
        parts = [
            f"{'═' * left}{iter_label}{'═' * right}\n",
            f" collect {collect_time:6.3f}s  learn {learn_time:6.3f}s  │  {loss_str}\n",
            f" reward {mean_rew}  eplen {mean_len}  │  noise {action_std.mean().item():5.2f}  │  {self.tot_timesteps} steps  {fps} fps\n",
        ]

        # Template-based extras rendering (hot path: value substitution only)
        for kind, *args in self._template:
            if kind == "s":
                parts.append(args[0])
            else:  # kind == "v"
                fmt, keys = args[0], args[1]
                parts.append(fmt.format(*[extras_dict.get(k, 0.0) for k in keys]))

        # Dynamic footer (timing changes)
        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        parts.extend([
            f" iter {iteration_time:5.2f}s  │  elapsed {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}  │  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}\n",
            f"{'═' * width}\n",
        ])

        print("".join(parts))
        self.ep_extras.clear()

        # Flush file logger if configured (periodic flushing for efficiency)
        if hasattr(self, '_runner') and hasattr(self._runner, 'file_logger'):
            flush_interval = getattr(self._runner, 'file_log_flush_interval', 10)
            if it % flush_interval == 0:
                self._runner.file_logger.flush()

