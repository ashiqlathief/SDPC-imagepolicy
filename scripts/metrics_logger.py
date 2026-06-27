
import os
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepData:
    pos: np.ndarray          # (3,) xyz
    action: np.ndarray       # (3,) delta-pos action taken


@dataclass
class EpisodeResult:
    # metadata
    encoder_type:       str
    latent_dim:         int
    horizon:            int
    n_diffusion_steps:  int
    num_candidates:     int
    variant:            str
    episode:            int
    seed:               int
    timestamp:          str

    # outcome
    success:            bool
    fell:               bool

    # progress through corridor
    steps_taken:        int
    final_x:            float   # x position at episode end
    max_x_reached:      float   # furthest x point ever reached
    x_progress_pct:     float   # max_x / 4.0 * 100

    # lateral behaviour
    mean_y_deviation:   float   # mean |y| — how much drone drifts sideways
    max_y_deviation:    float   # max |y|

    # altitude
    mean_z:             float   # should stay ~0.3
    std_z:              float   # how much z fluctuates

    # trajectory quality
    path_length:        float   # total distance traveled (sum of step distances)
    mean_smoothness:    float   # mean |a[t] - a[t-1]| — lower = smoother

    # safety — broken down by violation type
    corridor_violations:  int   # total = wall + halfspace + obstacle
    wall_violations:      int   # steps where |y| > 1.0
    hs_violations:        int   # steps crossing the diagonal halfspace line
    obstacle_violations:  int   # steps inside any box or cylinder footprint


# ─────────────────────────────────────────────────────────────────────────────
# Main logger class
# ─────────────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Records per-step data during episodes and computes metrics at episode end.
    Saves results to CSV (one row per episode) and JSON summary per run.
    """

    CORRIDOR_Y_LIMIT = 1.0   # |y| beyond this = violation
    GATE_X            = 4.0  # drone must reach this x to succeed

    BOX_RADIUS   = 0.20   # half-width of box obstacles (matches evalcopy)
    CYL_RADIUS   = 0.06   # radius of cylinder obstacles
    DRONE_RADIUS = 0.05   # approximate crazyflie body radius

    def __init__(
        self,
        save_dir:            str  = "results",
        encoder_type:        str  = "cnn",
        latent_dim:          int  = 256,
        horizon:             int  = 16,
        n_diffusion_steps:   int  = 20,
        num_candidates:      int  = 1,
        corridor_halfspaces: list = None,
        boxes:               list = None,   # list of (x, y) tuples
        cylinders:           list = None,   # list of (x, y) tuples
    ):
        self.save_dir            = save_dir
        self.encoder_type        = encoder_type
        self.latent_dim          = latent_dim
        self.horizon             = horizon
        self.n_diffusion_steps   = n_diffusion_steps
        self.num_candidates      = num_candidates
        self.corridor_halfspaces = corridor_halfspaces or []
        self.boxes               = boxes     or []
        self.cylinders           = cylinders or []

        os.makedirs(save_dir, exist_ok=True)

        # current episode state
        self._variant:      str             = ""
        self._episode:      int             = 0
        self._seed:         int             = 0
        self._steps:        List[StepData]  = []
        self._active:       bool            = False

        # all completed episodes
        self.results: List[EpisodeResult] = []

        print(f"[MetricsLogger] Saving to: {os.path.abspath(save_dir)}")
        print(f"[MetricsLogger] Config: encoder={encoder_type} latent={latent_dim} "
              f"H={horizon} K={num_candidates} T={n_diffusion_steps}")

    # ── Episode lifecycle ────────────────────────────────────────────────────

    def begin_episode(self, variant: str, episode: int = 0, seed: int = 0):
        """Call at the start of each episode."""
        if self._active:
            print("[MetricsLogger] WARNING: begin_episode called without end_episode.")
        self._variant  = variant
        self._episode  = episode
        self._seed     = seed
        self._steps    = []
        self._active   = True

    def step(self, pos: np.ndarray, action: np.ndarray):
        """
        Call at every control step.
        pos:    (3,) current xyz position
        action: (3,) action taken this step (delta-pos, real units)
        """
        if not self._active:
            return
        self._steps.append(StepData(
            pos=np.array(pos, dtype=np.float32),
            action=np.array(action, dtype=np.float32),
        ))

    def end_episode(self, success: bool, fell: bool):
        """Call at the end of each episode."""
        if not self._active:
            print("[MetricsLogger] WARNING: end_episode called without begin_episode.")
            return

        result = self._compute_metrics(success, fell)
        self.results.append(result)
        self._active = False

        # live print
        print(
            f"[Episode {result.episode:03d}] variant={result.variant:25s} "
            f"success={result.success} fell={result.fell} "
            f"steps={result.steps_taken:4d} "
            f"max_x={result.max_x_reached:.3f} ({result.x_progress_pct:.1f}%) "
            f"wall={result.wall_violations} "
            f"hs={result.hs_violations} "
            f"obs={result.obstacle_violations} "
            f"total_viol={result.corridor_violations}"
        )

    # ── Metric computation ───────────────────────────────────────────────────

    def _compute_metrics(self, success: bool, fell: bool) -> EpisodeResult:
        steps = self._steps
        n = len(steps)

        if n == 0:
            # empty episode — return zeros
            return self._zero_result(success, fell)

        positions = np.stack([s.pos    for s in steps], axis=0)  # (N, 3)
        actions   = np.stack([s.action for s in steps], axis=0)  # (N, 3)

        # ── corridor progress ──
        xs = positions[:, 0]
        final_x     = float(xs[-1])
        max_x       = float(xs.max())
        x_pct       = min(max_x / self.GATE_X * 100.0, 100.0)

        # ── lateral deviation ──
        ys              = np.abs(positions[:, 1])
        mean_y_dev      = float(ys.mean())
        max_y_dev       = float(ys.max())

        # ── altitude ──
        zs      = positions[:, 2]
        mean_z  = float(zs.mean())
        std_z   = float(zs.std())

        # ── path length (sum of euclidean step distances) ──
        if n > 1:
            diffs       = np.diff(positions, axis=0)          # (N-1, 3)
            step_dists  = np.linalg.norm(diffs, axis=1)       # (N-1,)
            path_length = float(step_dists.sum())
        else:
            path_length = 0.0

        # ── smoothness (mean |a[t] - a[t-1]|) ──
        if n > 1:
            action_diffs    = np.diff(actions, axis=0)                    # (N-1, 3)
            smoothness_vals = np.linalg.norm(action_diffs, axis=1)        # (N-1,)
            mean_smoothness = float(smoothness_vals.mean())
        else:
            mean_smoothness = 0.0

        xs_traj = positions[:, 0]
        ys_traj = positions[:, 1]

        # ── 1. Wall violations: |y| > 1.0 ───────────────────────────────────
        wall_violations = int((np.abs(ys_traj) > self.CORRIDOR_Y_LIMIT).sum())

        # ── 2. Halfspace violations: drone crosses diagonal constraint line ──
        hs_violations = 0
        for hs in self.corridor_halfspaces:
            p1, p2, side = hs
            p1x, p1y = p1
            p2x, p2y = p2
            m_hs    = (p2y - p1y) / (p2x - p1x)
            b_hs    = p1y - m_hs * p1x
            hs_line = m_hs * xs_traj + b_hs
            if side == 'below':
                hs_violations += int((ys_traj > hs_line).sum())  # above line = violation
            else:
                hs_violations += int((ys_traj < hs_line).sum())  # below line = violation

        # ── 3. Obstacle violations: drone enters obstacle footprint (XY) ────
        obstacle_violations = 0
        box_margin = self.BOX_RADIUS + self.DRONE_RADIUS   # 0.25 m
        cyl_margin = self.CYL_RADIUS + self.DRONE_RADIUS   # 0.11 m

        for (bx, by) in self.boxes:
            inside_x = np.abs(xs_traj - bx) < box_margin
            inside_y = np.abs(ys_traj - by) < box_margin
            obstacle_violations += int((inside_x & inside_y).sum())

        for (cx, cy) in self.cylinders:
            dist = np.sqrt((xs_traj - cx)**2 + (ys_traj - cy)**2)
            obstacle_violations += int((dist < cyl_margin).sum())

        violations = wall_violations + hs_violations + obstacle_violations

        return EpisodeResult(
            encoder_type        = self.encoder_type,
            latent_dim          = self.latent_dim,
            horizon             = self.horizon,
            n_diffusion_steps   = self.n_diffusion_steps,
            num_candidates      = self.num_candidates,
            variant             = self._variant,
            episode             = self._episode,
            seed                = self._seed,
            timestamp           = time.strftime("%Y-%m-%d %H:%M:%S"),

            success             = success,
            fell                = fell,

            steps_taken         = n,
            final_x             = final_x,
            max_x_reached       = max_x,
            x_progress_pct      = x_pct,

            mean_y_deviation    = mean_y_dev,
            max_y_deviation     = max_y_dev,

            mean_z              = mean_z,
            std_z               = std_z,

            path_length         = path_length,
            mean_smoothness     = mean_smoothness,

            corridor_violations  = violations,
            wall_violations      = wall_violations,
            hs_violations        = hs_violations,
            obstacle_violations  = obstacle_violations,
        )

    def _zero_result(self, success: bool, fell: bool) -> EpisodeResult:
        return EpisodeResult(
            encoder_type=self.encoder_type, latent_dim=self.latent_dim,
            horizon=self.horizon, n_diffusion_steps=self.n_diffusion_steps,
            num_candidates=self.num_candidates, variant=self._variant,
            episode=self._episode, seed=self._seed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            success=success, fell=fell, steps_taken=0,
            final_x=0.0, max_x_reached=0.0, x_progress_pct=0.0,
            mean_y_deviation=0.0, max_y_deviation=0.0,
            mean_z=0.0, std_z=0.0,
            path_length=0.0, mean_smoothness=0.0,
            corridor_violations=0,
            wall_violations=0,
            hs_violations=0,
            obstacle_violations=0,
        )

    # ── Save results ─────────────────────────────────────────────────────────

    def save(self):
        """Save all recorded episodes to results/summary.json."""
        if not self.results:
            print("[MetricsLogger] No results to save.")
            return

        self._save_summary()

    def _save_summary(self):
        """Compute mean ± std for each variant and save to JSON."""
        from collections import defaultdict

        # group by variant
        by_variant = defaultdict(list)
        for r in self.results:
            by_variant[r.variant].append(r)

        summary = {}
        for variant, episodes in by_variant.items():
            n = len(episodes)

            def mean(key):
                return float(np.mean([getattr(e, key) for e in episodes]))

            def std(key):
                return float(np.std([getattr(e, key) for e in episodes]))

            def rate(key):
                return float(np.mean([float(getattr(e, key)) for e in episodes]) * 100)

            summary[variant] = {
                "n_episodes":               n,
                "success_rate_%":           rate("success"),
                "collision_rate_%":         rate("fell"),
                "mean_steps":               mean("steps_taken"),
                "std_steps":                std("steps_taken"),
                "mean_max_x":               mean("max_x_reached"),
                "std_max_x":                std("max_x_reached"),
                "mean_x_progress_%":        mean("x_progress_pct"),
                "mean_y_deviation":         mean("mean_y_deviation"),
                "mean_corridor_violations":  mean("corridor_violations"),
                "mean_wall_violations":       mean("wall_violations"),
                "mean_hs_violations":         mean("hs_violations"),
                "mean_obstacle_violations":   mean("obstacle_violations"),
                "mean_altitude":            mean("mean_z"),
                "std_altitude":             std("mean_z"),
                "mean_path_length":         mean("path_length"),
                "mean_smoothness":          mean("mean_smoothness"),
            }

        # overall summary across all variants
        all_eps = self.results
        summary["_overall"] = {
            "encoder_type":     self.encoder_type,
            "latent_dim":       self.latent_dim,
            "horizon":          self.horizon,
            "n_diffusion_steps":self.n_diffusion_steps,
            "num_candidates":   self.num_candidates,
            "total_episodes":   len(all_eps),
            "overall_success_%": float(np.mean([float(e.success) for e in all_eps]) * 100),
        }

        json_path = os.path.join(self.save_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[MetricsLogger] Summary saved → {json_path}")
        print("\n── Summary ──────────────────────────────────────────")
        for variant, stats in summary.items():
            if variant == "_overall":
                continue
            print(f"  {variant:30s} "
                  f"success={stats['success_rate_%']:5.1f}%  "
                  f"fell={stats['collision_rate_%']:5.1f}%  "
                  f"max_x={stats['mean_max_x']:.3f}  "
                  f"progress={stats['mean_x_progress_%']:.1f}%  "
                  f"violations={stats['mean_corridor_violations']:.1f}")
        print("─────────────────────────────────────────────────────\n")

    def print_live_summary(self):
        """Print a quick summary of results so far — call anytime."""
        if not self.results:
            print("[MetricsLogger] No results yet.")
            return
        successes   = sum(1 for r in self.results if r.success)
        falls       = sum(1 for r in self.results if r.fell)
        n           = len(self.results)
        mean_x      = np.mean([r.max_x_reached for r in self.results])
        print(f"[Live] {n} episodes | success={successes/n*100:.1f}% | "
              f"fell={falls/n*100:.1f}% | mean_max_x={mean_x:.3f}")