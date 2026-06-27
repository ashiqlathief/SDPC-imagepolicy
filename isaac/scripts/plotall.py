"""
plotall.py  –  Plot all safety-variant trajectories on one XY figure.

Groups files by experiment (encoder + latent_dim). Each experiment gets its
own figure with all its variants overlaid.

Each .npz file contains:
    xyz            (T,3)   executed XYZ positions
    variant        str     e.g. "diffuser", "sdpc-c-tightened"
    encoder        str     e.g. "vitp", "cnn", "vit"
    latent_dim     int     e.g. 256, 512, 1024
    boxes          (N,2)   box obstacle centres (x,y)
    cylinders      (M,2)   cylinder obstacle centres (x,y)
    tighten        float   constraint tightening margin
    use_projection bool
    projection_mode str
    num_candidates int
    selection      str
    success        bool
    fell           bool

Usage
-----
python plotall.py --npz_dir plots/11/
python plotall.py --files traj_vitp_L512_diffuser.npz traj_vitp_L512_sdpc-c.npz
python plotall.py --demo
"""

import argparse
import glob
import importlib
import os
import sys
import types
from collections import defaultdict
import numpy as np

# ── load config (mock heavy deps) ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_du = types.ModuleType("diffuser.utils")
_du.watch = lambda *a, **kw: (lambda fn: fn)
sys.modules.setdefault("diffuser",       types.ModuleType("diffuser"))
sys.modules.setdefault("diffuser.utils", _du)
_cfg = importlib.import_module("config.avoiding-crazyflie")
CONFIG_HALFSPACES = list(getattr(_cfg, "CORRIDOR_HALFSPACES", []))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

# ── variant display names ─────────────────────────────────────────────────────
VARIANT_LABELS = {
    "diffuser":                  "Diffuser (no safety)",
    "sdpc-r":                    "SDPC-R",
    "sdpc-r-tightened":          "SDPC-R (tightened)",
    "sdpc-c":                    "SDPC-C",
    "sdpc-c-tightened":          "SDPC-C (tightened)",
    "sdpc-t":                    "SDPC-T",
    "sdpc-t-tightened":          "SDPC-T (tightened)",
    "gradient":                  "Gradient",
    "gradient-tightened":        "Gradient (tightened)",
    "post_processing":           "Post-proc.",
    "post_processing-tightened": "Post-proc. (tightened)",
    "model_free":                "Model-free",
    "model_free-tightened":      "Model-free (tightened)",
}

GROUP_COLORS = {
    "diffuser":        "#888888",
    "sdpc-r":          "#e6194b",
    "sdpc-c":          "#3cb44b",
    "sdpc-t":          "#4363d8",
    "gradient":        "#f58231",
    "post_processing": "#911eb4",
    "model_free":      "#42d4f4",
}

def _base_group(name: str) -> str:
    for g in GROUP_COLORS:
        if name.startswith(g):
            return g
    return name

def _line_style(name: str) -> str:
    return "--" if "tightened" in name else "-"

def _line_alpha(name: str) -> float:
    return 0.65 if "tightened" in name else 0.90


# ── obstacle drawing ──────────────────────────────────────────────────────────
def add_obstacles(ax, boxes, cylinders, tighten=0.0,
                  box_size=0.20, cyl_radius=0.06):
    for x, y in boxes:
        ax.add_patch(Rectangle(
            (x - box_size/2, y - box_size/2), box_size, box_size,
            linewidth=0.8, edgecolor="black", facecolor="#cc3030",
            alpha=0.35, zorder=2,
        ))
        if tighten > 0:
            m = tighten
            ax.add_patch(Rectangle(
                (x - box_size/2 - m, y - box_size/2 - m),
                box_size + 2*m, box_size + 2*m,
                linewidth=0.8, edgecolor="#cc3030", facecolor="none",
                linestyle=":", alpha=0.55, zorder=2,
            ))
    for x, y in cylinders:
        ax.add_patch(Circle(
            (x, y), cyl_radius,
            linewidth=0.8, edgecolor="black", facecolor="#e87020",
            alpha=0.40, zorder=2,
        ))
        if tighten > 0:
            ax.add_patch(Circle(
                (x, y), cyl_radius + tighten,
                linewidth=0.8, edgecolor="#e87020", facecolor="none",
                linestyle=":", alpha=0.55, zorder=2,
            ))


def add_dynamic_bands(ax, cylinders, dynamic_cyl_indices, obs_amplitude, obs_axes,
                      cyl_radius=0.06,
                      x_clamp=(-0.5, 4.5), y_clamp=(-1.0, 1.0)):
    """Draw oscillation sweep bands behind dynamic cylinders."""
    if len(dynamic_cyl_indices) == 0:
        return
    for k, idx in enumerate(dynamic_cyl_indices):
        x0, y0 = cylinders[int(idx)]
        # broadcast single-element obs_axes to all dynamic cylinders
        axis = obs_axes[k] if k < len(obs_axes) else (obs_axes[0] if obs_axes else "y")
        if axis == "y":
            ylo = max(y_clamp[0], y0 - obs_amplitude)
            yhi = min(y_clamp[1], y0 + obs_amplitude)
            band = Rectangle((x0 - cyl_radius, ylo), 2*cyl_radius, yhi - ylo,
                              facecolor="tab:orange", alpha=0.22, linewidth=0, zorder=1)
        elif axis == "x":
            xlo = max(x_clamp[0], x0 - obs_amplitude)
            xhi = min(x_clamp[1], x0 + obs_amplitude)
            band = Rectangle((xlo, y0 - cyl_radius), xhi - xlo, 2*cyl_radius,
                              facecolor="tab:orange", alpha=0.22, linewidth=0, zorder=1)
        else:  # "xy" — same delta applied to both axes → 45° diagonal line sweep
            x_lo = max(x_clamp[0], x0 - obs_amplitude)
            x_hi = min(x_clamp[1], x0 + obs_amplitude)
            y_lo = max(y_clamp[0], y0 - obs_amplitude)
            y_hi = min(y_clamp[1], y0 + obs_amplitude)
            p1 = np.array([x_lo, y_lo])
            p2 = np.array([x_hi, y_hi])
            seg = p2 - p1
            d_hat = seg / (np.linalg.norm(seg) + 1e-8)
            n_hat = np.array([-d_hat[1], d_hat[0]])
            corners = np.array([p1 - cyl_radius*n_hat, p1 + cyl_radius*n_hat,
                                 p2 + cyl_radius*n_hat, p2 - cyl_radius*n_hat])
            band = Polygon(corners, facecolor="tab:orange", alpha=0.15, zorder=1)
        ax.add_patch(band)


# ── load one npz ─────────────────────────────────────────────────────────────
def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)

    xyz       = d["xyz"].astype(np.float32)
    variant   = str(d["variant"])   if "variant"   in d else os.path.basename(path)
    encoder   = str(d["encoder"])   if "encoder"   in d else "unknown"
    latent_dim= int(d["latent_dim"])if "latent_dim" in d else 0
    success   = bool(d["success"])  if "success"   in d else None
    fell      = bool(d["fell"])     if "fell"      in d else None
    episode   = int(d["episode"])   if "episode"   in d else 0

    boxes     = np.array(d["boxes"],     dtype=np.float32) if "boxes"     in d else np.zeros((0,2),np.float32)
    cylinders = np.array(d["cylinders"], dtype=np.float32) if "cylinders" in d else np.zeros((0,2),np.float32)

    tighten        = float(d["tighten"])         if "tighten"         in d else 0.0
    use_projection = bool(d["use_projection"])   if "use_projection"  in d else False
    projection_mode= str(d["projection_mode"])   if "projection_mode" in d else "none"
    num_candidates = int(d["num_candidates"])    if "num_candidates"  in d else 1
    selection      = str(d["selection"])         if "selection"       in d else "first"

    dynamic_obstacles   = bool(d["dynamic_obstacles"])                        if "dynamic_obstacles"   in d else False
    dynamic_cyl_indices = np.array(d["dynamic_cyl_indices"], dtype=np.int32)  if "dynamic_cyl_indices" in d else np.array([], dtype=np.int32)
    obs_amplitude       = float(d["obs_amplitude"])                           if "obs_amplitude"       in d else 0.35
    obs_frequency       = float(d["obs_frequency"])                           if "obs_frequency"       in d else 0.25
    obs_axes            = list(d["obs_axes"])                                  if "obs_axes"            in d else []

    floating_spheres = bool(d["floating_spheres"])                                   if "floating_spheres"  in d else False
    sphere_positions = np.array(d["sphere_positions"], dtype=np.float32)             if "sphere_positions"  in d else np.zeros((0, 3), np.float32)
    sphere_radius    = float(d["sphere_radius"])                                      if "sphere_radius"     in d else 0.10
    sphere_amplitude = float(d["sphere_amplitude"])                                   if "sphere_amplitude"  in d else 0.20
    sph_xyz_traj     = np.array(d["sph_xyz_traj"], dtype=np.float32)                 if "sph_xyz_traj"      in d else np.zeros((0, 0, 3), np.float32)

    # only plot what was active during the recorded run
    if "halfspaces" in d and len(d["halfspaces"]) > 0:
        hs_pts   = d["halfspaces"].tolist()
        hs_sides = d["hs_sides"].tolist() if "hs_sides" in d else ["below"] * len(hs_pts)
        halfspaces = [[pts[0], pts[1], side] for pts, side in zip(hs_pts, hs_sides)]
    else:
        halfspaces = []

    return dict(
        xyz=xyz, variant=variant, encoder=encoder, latent_dim=latent_dim,
        success=success, fell=fell, episode=episode,
        boxes=boxes, cylinders=cylinders,
        tighten=tighten, use_projection=use_projection,
        projection_mode=projection_mode,
        num_candidates=num_candidates, selection=selection,
        dynamic_obstacles=dynamic_obstacles,
        dynamic_cyl_indices=dynamic_cyl_indices,
        obs_amplitude=obs_amplitude,
        obs_frequency=obs_frequency,
        obs_axes=obs_axes,
        floating_spheres=floating_spheres,
        sphere_positions=sphere_positions,
        sphere_radius=sphere_radius,
        sphere_amplitude=sphere_amplitude,
        sph_xyz_traj=sph_xyz_traj,
        halfspaces=halfspaces,
    )


# ── halfspace overlay ─────────────────────────────────────────────────────────
def plot_halfspace_constraints_xy(ax, halfspaces, xlim, ylim, alpha=0.10):
    """Draw boundary lines and shade the infeasible half-plane for each halfspace."""
    xmin, xmax = xlim
    ymin, ymax = ylim
    for hs in halfspaces:
        p1, p2, side = hs
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 1e-8:
            ax.axvline(x1, color="royalblue", linewidth=1.8, alpha=0.7, linestyle="--")
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        xs = np.array([xmin, xmax], dtype=np.float64)
        ys = m * xs + b
        ax.plot(xs, ys, color="royalblue", linewidth=1.8, alpha=0.7, linestyle="--")
        if side == "above":
            ax.fill_between(xs, ys, ymin, color="royalblue", alpha=alpha)
        else:
            ax.fill_between(xs, ys, ymax, color="royalblue", alpha=alpha)


# ── plot one experiment (all variants on one figure) ─────────────────────────
def plot_experiment(
    records: list,            # list of load_npz dicts, all same encoder+latent_dim
    out_path: str = None,
    show: bool = True,
    plane: str = "xy",        # "xy" top-down  |  "xz" side / altitude view
):
    # ── axis mapping ──────────────────────────────────────────────────────────
    if plane == "xz":
        h_idx, v_idx = 0, 2
        h_label, v_label = "x  (m)", "z  (m)"
        h_lim = (-0.2, 4.7)
        v_lim = (-0.05, 1.35)
    else:
        h_idx, v_idx = 0, 1
        h_label, v_label = "x  (m)", "y  (m)"
        h_lim = (-0.2, 4.7)
        v_lim = (-1.2, 1.2)

    enc = records[0]["encoder"]
    ldim = records[0]["latent_dim"]
    exp_label = f"{enc.upper()}  –  latent dim {ldim}"

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    ref = records[0]
    corridor_end = 4.1

    wall_kw = dict(linewidth=1.5, edgecolor="#555555", facecolor="#cccccc",
                   alpha=0.50, zorder=1)

    # ── static scene geometry (plane-aware) ───────────────────────────────────
    if plane == "xy":
        ax.add_patch(Rectangle((0.0,  1.00), corridor_end, 0.10, **wall_kw))
        ax.add_patch(Rectangle((0.0, -1.10), corridor_end, 0.10, **wall_kw))
        ax.add_patch(Rectangle((corridor_end, -1.10), 0.10, 2.20, **wall_kw))
    else:  # xz: floor and ceiling lines
        ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.55)
        ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.55)

    # ── obstacles (plane-aware) ────────────────────────────────────────────────
    if plane == "xy":
        if ref.get("dynamic_obstacles"):
            add_dynamic_bands(ax, ref["cylinders"], ref["dynamic_cyl_indices"],
                              ref["obs_amplitude"], ref["obs_axes"])
        add_obstacles(ax, ref["boxes"], ref["cylinders"], tighten=0.0)
    else:  # xz: cylinders → full-height vertical strips; boxes similarly
        for x, y in ref["boxes"]:
            ax.add_patch(Rectangle(
                (x - 0.10, 0.0), 0.20, 1.0,
                linewidth=0.8, edgecolor="black", facecolor="#cc3030",
                alpha=0.22, zorder=2,
            ))
        for x, y in ref["cylinders"]:
            ax.add_patch(Rectangle(
                (x - 0.06, 0.0), 0.12, 1.0,
                linewidth=0.8, edgecolor="black", facecolor="#e87020",
                alpha=0.25, zorder=2,
            ))

    # ── floating spheres ───────────────────────────────────────────────────────
    if ref.get("floating_spheres") and len(ref["sphere_positions"]) > 0:
        sr  = ref["sphere_radius"]
        amp = ref["sphere_amplitude"]
        for (sx, sy, sz) in ref["sphere_positions"]:
            hc = sx
            vc = sy if plane == "xy" else sz
            ax.add_patch(Circle(
                (hc, vc), sr,
                linewidth=0.8, edgecolor="#1f6fbf", facecolor="#4a9de0",
                alpha=0.40, zorder=2,
            ))
            ax.add_patch(Circle(
                (hc, vc), sr + amp,
                linewidth=0.8, edgecolor="#4a9de0", facecolor="none",
                linestyle=":", alpha=0.35, zorder=1,
            ))

    # ── halfspace constraints (XY only) ──────────────────────────────────────
    halfspaces = ref.get("halfspaces", [])
    if plane == "xy" and halfspaces:
        plot_halfspace_constraints_xy(ax, halfspaces, xlim=h_lim, ylim=v_lim)

    seen_tighten = set()
    for rec in records:
        t = rec["tighten"]
        if t > 0 and t not in seen_tighten:
            seen_tighten.add(t)
            if plane == "xy":
                for x, y in rec["boxes"]:
                    ax.add_patch(Circle(
                        (x, y), 0.15 + t,
                        linewidth=0.9, edgecolor="#cc3030", facecolor="none",
                        linestyle=":", alpha=0.55, zorder=2,
                    ))
                for x, y in rec["cylinders"]:
                    ax.add_patch(Circle(
                        (x, y), 0.06 + t,
                        linewidth=0.9, edgecolor="#e87020", facecolor="none",
                        linestyle=":", alpha=0.55, zorder=2,
                    ))

    # ── trajectories ──────────────────────────────────────────────────────────
    legend_handles = []
    sorted_recs = sorted(records, key=lambda r: (
        r["variant"].replace("-tightened", ""), "tightened" in r["variant"]))

    for rec in sorted_recs:
        variant_name = rec["variant"]
        xyz = rec["xyz"]
        if xyz is None or len(xyz) == 0:
            continue

        hs = xyz[:, h_idx]
        vs = xyz[:, v_idx]

        base  = _base_group(variant_name)
        color = GROUP_COLORS.get(base, "#aaaaaa")
        ls    = _line_style(variant_name)
        alpha = _line_alpha(variant_name)
        lw    = 1.6 if "tightened" in variant_name else 2.2
        label = VARIANT_LABELS.get(variant_name, variant_name)

        if rec["success"] is not None:
            badge = " ✓" if rec["success"] else (" ✗ fell" if rec["fell"] else " ✗")
            label = label + badge

        ax.plot(hs, vs, color=color, linestyle=ls,
                linewidth=lw, alpha=alpha, zorder=3)
        ax.scatter(hs[0],  vs[0],  color=color, s=28, zorder=4, alpha=alpha)
        ax.scatter(hs[-1], vs[-1], color=color, s=40, marker="x",
                   linewidths=1.2, zorder=4, alpha=alpha)

        legend_handles.append(
            Line2D([0], [0], color=color, linestyle=ls, linewidth=lw,
                   alpha=alpha, label=label)
        )

    # ── goal line ─────────────────────────────────────────────────────────────
    if plane == "xy":
        ax.plot([4.0, 4.0], [-1.0, 1.0], color="#00aa00", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=5)
    else:
        ax.plot([4.0, 4.0], [0.0, 1.0], color="#00aa00", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=5)
    legend_handles.append(
        Line2D([0], [0], color="#00aa00", linewidth=2.2, label="Goal (x = 4 m)")
    )

    # ── obstacle legend proxies ───────────────────────────────────────────────
    legend_handles += [
        Rectangle((0,0),1,1, facecolor="#cc3030", alpha=0.4,
                  edgecolor="black", linewidth=0.8, label="Box obstacle"),
        Circle((0,0),1, facecolor="#e87020", alpha=0.45,
               edgecolor="black", linewidth=0.8, label="Cylinder obstacle"),
    ]
    if ref.get("dynamic_obstacles"):
        amp = ref.get("obs_amplitude", 0.35)
        legend_handles.append(
            Rectangle((0,0), 1, 0.5, facecolor="tab:orange", alpha=0.22,
                      linewidth=0, label=f"Oscillation range (±{amp:.2g} m)")
        )
    if ref.get("floating_spheres") and len(ref["sphere_positions"]) > 0:
        sr  = ref["sphere_radius"]
        amp = ref["sphere_amplitude"]
        legend_handles.append(
            Circle((0, 0), 1, facecolor="#4a9de0", alpha=0.40,
                   edgecolor="#1f6fbf", linewidth=0.8,
                   label=f"Floating sphere (r={sr} m)")
        )
        legend_handles.append(
            Line2D([0], [0], color="#4a9de0", linewidth=0.8, linestyle=":",
                   alpha=0.55, label=f"Sphere xy sweep (±{amp:.2g} m)")
        )
    if seen_tighten:
        t_str = ", ".join(f"{t:.3g} m" for t in sorted(seen_tighten))
        legend_handles.append(
            Line2D([0],[0], color="#777777", linestyle=":", linewidth=1.2,
                   label=f"Constraint margin ({t_str})")
        )
    if plane == "xy" and halfspaces:
        from matplotlib.patches import Patch
        legend_handles.append(
            Patch(facecolor="royalblue", alpha=0.3, label="Halfspace constraint")
        )

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(h_label, fontsize=11)
    ax.set_ylabel(v_label, fontsize=11)
    plane_tag = "Top-down (XY)" if plane == "xy" else "Side view (XZ — altitude)"
    ax.set_title(f"Safety Variant Trajectories — {exp_label}  [{plane_tag}]",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")  # shrinks axes box, preserves data limits
    ax.set_xlim(*h_lim)
    ax.set_ylim(*v_lim)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=8.5,
        framealpha=0.9,
        title="Variant  (✓=success  ✗=fail)",
        title_fontsize=9,
    )

    fig.tight_layout(rect=[0, 0, 0.76, 1])

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


# ── demo synthetic data ───────────────────────────────────────────────────────
def _make_demo_data():
    rng = np.random.default_rng(0)
    xs = np.linspace(0.05, 4.2, 200)

    boxes_A = np.array([(3.5,0.2),(2.2,-0.2),(1.6,0.2),(3.7,-0.3),(2.8,-0.2),(1.0,-0.3)], dtype=np.float32)
    cyls_A  = np.array([(1.4,-0.2),(2.8,0.3),(1.2,0.4),(3.5,0.4),(2.0,0.4),(2.5,0.3)],   dtype=np.float32)

    # Experiment B: shifted obstacles (different env layout)
    boxes_B = boxes_A + np.array([0.2, 0.05], dtype=np.float32)
    cyls_B  = cyls_A  + np.array([-0.1, 0.05], dtype=np.float32)

    experiments = [
        ("vitp", 512, boxes_A, cyls_A),
        ("cnn",  256, boxes_B, cyls_B),
    ]

    # demo sphere data — 4 spheres at varied heights
    _demo_sph_pos = np.array([
        (0.8, -0.20, 0.50), (1.6,  0.30, 0.70),
        (2.5,  0.00, 0.35), (3.4, -0.25, 0.65),
    ], dtype=np.float32)
    _demo_sph_r   = 0.10
    _demo_sph_amp = 0.20
    N_sph = len(_demo_sph_pos)
    _demo_sph_xyz_traj = np.stack([
        np.column_stack([
            np.full(len(xs), _demo_sph_pos[k, 0]),
            np.full(len(xs), _demo_sph_pos[k, 1]),
            _demo_sph_pos[k, 2] + _demo_sph_amp * np.sin(
                2 * np.pi * 0.20 * 0.70 * xs + 2 * np.pi * k / N_sph
            ),
        ])
        for k in range(N_sph)
    ], axis=1).astype(np.float32)  # (200, N_sph, 3)

    all_records = defaultdict(list)
    for enc, ldim, boxes, cyls in experiments:
        for i, v in enumerate(VARIANT_LABELS.keys()):
            amp   = 0.25 + 0.15*(i%4)
            freq  = 0.8  + 0.40*(i%3)
            phase = rng.uniform(0, np.pi)
            ys = amp*np.sin(freq*xs+phase) + rng.normal(0,0.02,len(xs))
            zs = 0.3 + 0.2*np.sin(0.5*xs+phase)
            tighten = 0.05 if "tightened" in v else 0.0
            rec = dict(
                xyz=np.stack([xs,ys,zs],axis=1).astype(np.float32),
                variant=v, encoder=enc, latent_dim=ldim,
                success=rng.random()>0.3, fell=rng.random()<0.1,
                episode=i,
                boxes=boxes, cylinders=cyls,
                tighten=tighten,
                use_projection="tightened" in v,
                projection_mode="post" if v!="diffuser" else "none",
                num_candidates=8 if v.startswith("sdpc-c") or v.startswith("sdpc-t") else 1,
                selection="minimum_projection_cost" if v.startswith("sdpc-c") else "first",
                dynamic_obstacles=False,
                dynamic_cyl_indices=np.array([], dtype=np.int32),
                obs_amplitude=0.35, obs_frequency=0.25, obs_axes=[],
                floating_spheres=True,
                sphere_positions=_demo_sph_pos,
                sphere_radius=_demo_sph_r,
                sphere_amplitude=_demo_sph_amp,
                sph_xyz_traj=_demo_sph_xyz_traj,
                halfspaces=[],
            )
            all_records[(enc, ldim)].append(rec)

    return all_records


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", type=str, default=None,
                   help="Directory with traj_*.npz files (all experiments).")
    p.add_argument("--files",   nargs="*", default=None,
                   help="Explicit .npz file list.")
    p.add_argument("--demo",    action="store_true")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for per-experiment PDFs.")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--plane",   type=str, default="xy", choices=["xy", "xz"],
                   help="Projection plane: 'xy' top-down (default) or "
                        "'xz' side view showing altitude vs corridor progress.")
    args = p.parse_args()

    # ── collect records grouped by (encoder, latent_dim) ─────────────────────
    grouped = defaultdict(list)   # {(encoder, latent_dim): [rec, ...]}

    if args.demo:
        print("[INFO] Demo mode.")
        grouped = _make_demo_data()

    else:
        if args.npz_dir:
            files = sorted(glob.glob(os.path.join(args.npz_dir, "traj_*.npz")))
        elif args.files:
            files = args.files
        else:
            print("[INFO] No input — demo mode.")
            grouped = _make_demo_data()
            files = []

        for f in files:
            rec = load_npz(f)
            key = (rec["encoder"], rec["latent_dim"])
            grouped[key].append(rec)

        print(f"[INFO] Loaded {sum(len(v) for v in grouped.values())} files "
              f"across {len(grouped)} experiment(s).")

    # ── one figure per experiment ─────────────────────────────────────────────
    out_dir = args.out_dir or args.npz_dir or "."
    plane_suffix = f"_{args.plane}" if args.plane != "xy" else ""
    for (enc, ldim), records in sorted(grouped.items()):
        fname = f"all_variants_{enc}_L{ldim}{plane_suffix}.pdf"
        out_path = os.path.join(out_dir, fname)
        print(f"[INFO] Plotting {enc} L{ldim}  ({len(records)} variants) "
              f"[plane={args.plane}] -> {fname}")
        plot_experiment(records, out_path=out_path, show=not args.no_show,
                        plane=args.plane)


if __name__ == "__main__":
    main()