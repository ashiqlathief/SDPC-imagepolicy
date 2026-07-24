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
python plotall.py --dir plots/11/
python plotall.py --dir plots/11/ plots/12/ plots/13/
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
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch
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

ENCODER_DISPLAY = {
    "raw":  "Raw-Pixel Transformer",
    "vitp": "ViT+U-Net",
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


# matches scripts/eval_crazieflie.py:647 — hardcoded there too, no CLI override
DRONE_RADIUS = 0.08


# ── obstacle drawing ──────────────────────────────────────────────────────────
def add_obstacles(ax, boxes, cylinders, tighten=0.0,
                  box_size=0.20, cyl_radius=0.06, drone_radius=DRONE_RADIUS):
    # unsafe/exclusion zone = physical footprint expanded by the drone's own
    # bounding radius plus whatever extra margin ("tighten") the projector used
    m = drone_radius + tighten
    for x, y in boxes:
        ax.add_patch(Rectangle(
            (x - box_size/2, y - box_size/2), box_size, box_size,
            linewidth=0.8, edgecolor="black", facecolor="#cc3030",
            alpha=0.35, zorder=2,
        ))
        ax.add_patch(Rectangle(
            (x - box_size/2 - m, y - box_size/2 - m),
            box_size + 2*m, box_size + 2*m,
            linewidth=0.8, edgecolor="#cc3030", facecolor="#cc3030",
            linestyle=":", alpha=0.12, zorder=2,
        ))
    for x, y in cylinders:
        ax.add_patch(Circle(
            (x, y), cyl_radius,
            linewidth=0.8, edgecolor="black", facecolor="#e87020",
            alpha=0.40, zorder=2,
        ))
        ax.add_patch(Circle(
            (x, y), cyl_radius + m,
            linewidth=0.8, edgecolor="#e87020", facecolor="#e87020",
            linestyle=":", alpha=0.12, zorder=2,
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

    cand_traj_xy = np.array(d["cand_traj_xy"], dtype=np.float32) if "cand_traj_xy" in d else np.zeros((0, 0, 0, 2), np.float32)
    snap_chosen  = np.array(d["snap_chosen"],  dtype=np.int64)   if "snap_chosen"  in d else np.zeros((0,), np.int64)
    cyl_xy_traj  = np.array(d["cyl_xy_traj"],  dtype=np.float32) if "cyl_xy_traj"  in d else np.zeros((0, 0, 2), np.float32)
    # projected/safe horizon for the chosen candidate, added offline by
    cand_traj_xy_proj = np.array(d["cand_traj_xy_proj"], dtype=np.float32) if "cand_traj_xy_proj" in d else np.zeros((0, 0, 2), np.float32)

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
        cand_traj_xy=cand_traj_xy,
        cand_traj_xy_proj=cand_traj_xy_proj,
        snap_chosen=snap_chosen,
        cyl_xy_traj=cyl_xy_traj,
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


# XY_FIGSIZE is the reference canvas -- with equal-aspect + adjustable="box",
# its rendered box ends up height-bound (11in x 5.5in), giving a scale of
# ~2.619 in/m. The xz box is derived below to hit that same scale, instead of
# reusing XY_FIGSIZE directly: with a shorter v-span (altitude vs. corridor
# width), reusing the xy canvas would make the xz box width-bound instead,
# rendering 1m at a different (larger) size than in the xy plot -- the two
# companion panels would look inconsistently scaled next to each other.
XY_FIGSIZE = (12, 5.5)
XY_H_LIM, XY_V_LIM = (-0.05, 4.15), (-1.05, 1.05)
XZ_H_LIM, XZ_V_LIM = (-0.05, 4.15), (-0.05, 1.1)


def _panel_box_inches(plane):
    """Returns (box_w, box_h) in inches: the actual rendered size of one
    plot_experiment panel once set_aspect('equal', adjustable='box') has
    shrunk it to fit -- i.e. the size a containing figure/cell must supply
    for that panel to fill it exactly, with no leftover centering slack."""
    xy_h_span = XY_H_LIM[1] - XY_H_LIM[0]
    xy_v_span = XY_V_LIM[1] - XY_V_LIM[0]
    xy_required_aspect = xy_v_span / xy_h_span
    xy_canvas_aspect = XY_FIGSIZE[1] / XY_FIGSIZE[0]
    if xy_required_aspect > xy_canvas_aspect:
        xy_box_h, xy_box_w = XY_FIGSIZE[1], XY_FIGSIZE[1] / xy_required_aspect
    else:
        xy_box_w, xy_box_h = XY_FIGSIZE[0], XY_FIGSIZE[0] * xy_required_aspect
    if plane != "xz":
        return xy_box_w, xy_box_h
    scale = xy_box_w / xy_h_span  # == xy_box_h / xy_v_span, inches per metre
    h_span = XZ_H_LIM[1] - XZ_H_LIM[0]
    v_span = XZ_V_LIM[1] - XZ_V_LIM[0]
    return h_span * scale, v_span * scale


# ── plot one experiment (all variants on one figure) ─────────────────────────
def plot_experiment(
    records: list,            # list of load_npz dicts, all same encoder+latent_dim
    out_path: str = None,
    show: bool = True,
    plane: str = "xy",        # "xy" top-down  |  "xz" side / altitude view
    ax=None,                  # draw onto this existing Axes instead of making
                              # a standalone figure (used by plot_experiment_2x2);
                              # out_path/show are ignored when ax is given, since
                              # the caller owns that figure's save/show/close.
):
    # ── axis mapping ──────────────────────────────────────────────────────────
    if plane == "xz":
        h_idx, v_idx = 0, 2
        h_label, v_label = "X Position (m)", "Z Position (m)"
        h_lim, v_lim = XZ_H_LIM, XZ_V_LIM
    else:
        h_idx, v_idx = 0, 1
        h_label, v_label = "X Position (m)", "Y Position (m)"
        h_lim, v_lim = XY_H_LIM, XY_V_LIM

    # ── figure ────────────────────────────────────────────────────────────────
    figsize = _panel_box_inches(plane)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.figure

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
        # tighten=0.0 here draws the baseline (drone_radius-only) exclusion
        # zone shared by every overlaid variant; tightened variants add an
        # extra margin on top of this that isn't shown per-variant here
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
        Circle((0,0),1, facecolor="#e87020", alpha=0.12,
               edgecolor="#e87020", linestyle=":", linewidth=0.8,
               label=f"Unsafe/exclusion zone (drone_radius={DRONE_RADIUS} m)"),
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
    ax.set_xlabel(h_label, fontsize=18, fontweight="bold")
    ax.set_ylabel(v_label, fontsize=18, fontweight="bold")
    ax.set_title("")
    ax.set_aspect("equal", adjustable="box")  # shrinks axes box, preserves data limits
    ax.set_xlim(*h_lim)
    ax.set_ylim(*v_lim)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # ax.legend(
    #     handles=legend_handles,
    #     loc="upper left",
    #     bbox_to_anchor=(1.01, 1.0),
    #     borderaxespad=0,
    #     fontsize=8.5,
    #     framealpha=0.9,
    #     title="Variant  (✓=success  ✗=fail)",
    #     title_fontsize=9,
    # )

    if standalone:
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[SAVED] {out_path}")

        if show:
            plt.show()

        plt.close(fig)
    return fig


def plot_experiment_2x2(records_by_side, out_path=None, show=True,
                         labels=("Raw-Pixel Transformer", "ViT+U-Net")):
    """Combines two models' top-down (xy) and side (xz) views into one 2x2
    figure: xy row on top, xz row on bottom, records_by_side[0] on the left
    and records_by_side[1] on the right (matching the left=Transformer,
    right=ViT+U-Net convention used everywhere else in the thesis) -- the
    single-figure counterpart of manually combining a pair of Figs. 7.3/7.4
    panels in an image editor."""
    # set_aspect("equal", adjustable="box") in plot_experiment shrinks each
    # panel to fit whichever of its cell's width/height is the tighter
    # constraint, then centers it in the leftover space -- so unless the
    # figure's overall width:height matches the panels' own content aspect,
    # picking a figsize by feel (e.g. 16x9) leaves each panel visibly
    # centered inside a too-big cell, which shows up as dead space between
    # rows/columns no amount of hspace/wspace tuning can close. Deriving the
    # figsize from the same box sizes plot_experiment itself renders at
    # avoids that mismatch instead of fighting it after the fact.
    box_w, xy_box_h = _panel_box_inches("xy")
    _,     xz_box_h = _panel_box_inches("xz")
    label_margin_w = 1.3   # left-side y-axis label + tick numbers
    label_margin_h = 1.0   # title above row 0 + xlabel below row 1
    fig_w = 2 * box_w + label_margin_w
    fig_h = xy_box_h + xz_box_h + label_margin_h

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                              gridspec_kw={"height_ratios": [xy_box_h, xz_box_h]})
    fig.patch.set_facecolor("white")

    for col, records in enumerate(records_by_side):
        for row, plane in enumerate(["xy", "xz"]):
            plot_experiment(records, ax=axes[row][col], plane=plane)

    for col, label in enumerate(labels):
        axes[0][col].set_title(label, fontsize=20, fontweight="bold")

    # top row shares the same x-axis (X Position, 0-4.15 m) as the bottom row
    # right below it -- drop its tick labels/xlabel so that space isn't spent
    # twice on the same axis.
    for col in range(2):
        axes[0][col].set_xlabel("")
        axes[0][col].tick_params(labelbottom=False)

    # right column shares the same y-axis (Y/Z Position, same range) as the
    # left column -- drop its label and tick numbers entirely instead of
    # repeating them, then align what's left of the two rows' y-labels onto
    # one common vertical line instead of each being auto-placed off its own
    # tick-label width.
    for row in range(2):
        axes[row][1].set_ylabel("")
        axes[row][1].tick_params(labelleft=False)
    fig.align_ylabels(axes[:, 0])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── per-step plots (chosen + unchosen candidate trajectories) ─────────────────
def plot_variant_steps(
    rec: dict,
    out_dir: str,
    plane: str = "xy",
):
    """One PDF per environment step for a single variant, showing the executed
    trail so far plus (when available) the chosen candidate's predicted
    horizon: the raw diffusion output and the SLSQP-projected/safe version
    of that same horizon, side by side."""
    variant   = rec["variant"]
    xyz       = rec["xyz"]
    T         = len(xyz)
    K         = int(rec.get("num_candidates", 1))

    cand_traj_xy      = rec.get("cand_traj_xy", None)
    cand_traj_xy_proj = rec.get("cand_traj_xy_proj", None)
    snap_chosen       = rec.get("snap_chosen", None)
    has_horizon = (
        cand_traj_xy is not None and np.asarray(cand_traj_xy).size > 0
        and cand_traj_xy_proj is not None and np.asarray(cand_traj_xy_proj).size > 0
    )
    if has_horizon:
        cand_traj_xy      = np.asarray(cand_traj_xy)
        cand_traj_xy_proj = np.asarray(cand_traj_xy_proj)
        snap_chosen       = np.asarray(snap_chosen)

    cyl_xy_traj = rec.get("cyl_xy_traj", None)
    dyn_indices = rec.get("dynamic_cyl_indices", np.array([], dtype=np.int32))
    has_cyl_traj = (
        rec.get("dynamic_obstacles", False)
        and cyl_xy_traj is not None
        and np.asarray(cyl_xy_traj).size > 0
    )
    if has_cyl_traj:
        cyl_xy_traj = np.asarray(cyl_xy_traj)

    if plane == "xz":
        h_idx, v_idx = 0, 2
        h_label, v_label = "x  (m)", "z  (m)"
        h_lim, v_lim = (-0.2, 4.7), (-0.05, 1.35)
    else:
        h_idx, v_idx = 0, 1
        h_label, v_label = "x  (m)", "y  (m)"
        h_lim, v_lim = (-0.2, 4.7), (-1.2, 1.2)

    hs_all = xyz[:, h_idx]
    vs_all = xyz[:, v_idx]

    corridor_end = 4.1
    wall_kw = dict(linewidth=1.5, edgecolor="#555555", facecolor="#cccccc",
                   alpha=0.50, zorder=1)
    halfspaces = rec.get("halfspaces", [])

    variant_dir = os.path.join(out_dir, f"steps_{variant}")
    os.makedirs(variant_dir, exist_ok=True)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(12, 5.5))
        fig.patch.set_facecolor("white")

        # ── static scene ──────────────────────────────────────────────────────
        if plane == "xy":
            ax.add_patch(Rectangle((0.0,  1.00), corridor_end, 0.10, **wall_kw))
            ax.add_patch(Rectangle((0.0, -1.10), corridor_end, 0.10, **wall_kw))
            ax.add_patch(Rectangle((corridor_end, -1.10), 0.10, 2.20, **wall_kw))
            if halfspaces:
                plot_halfspace_constraints_xy(ax, halfspaces, xlim=h_lim, ylim=v_lim)
        else:
            ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.55)
            ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.55)

        # ── obstacles (cylinders at this step's snapshot if dynamic) ────────────
        cylinders_t = rec["cylinders"].copy()
        if has_cyl_traj:
            ci = min(max(t - 1, 0), cyl_xy_traj.shape[0] - 1)
            for idx in dyn_indices:
                cylinders_t[int(idx)] = cyl_xy_traj[ci, int(idx)]

        if plane == "xy":
            add_obstacles(ax, rec["boxes"], cylinders_t, tighten=rec.get("tighten", 0.0))
        else:
            for x, y in rec["boxes"]:
                ax.add_patch(Rectangle((x - 0.10, 0.0), 0.20, 1.0, linewidth=0.8,
                                        edgecolor="black", facecolor="#cc3030",
                                        alpha=0.22, zorder=2))
            for x, y in cylinders_t:
                ax.add_patch(Rectangle((x - 0.06, 0.0), 0.12, 1.0, linewidth=0.8,
                                        edgecolor="black", facecolor="#e87020",
                                        alpha=0.25, zorder=2))

        # ── moving-obstacle direction arrows (xy only) ───────────────────────────
        if plane == "xy" and has_cyl_traj and ci > 0:
            for idx in dyn_indices:
                idx = int(idx)
                cx, cy = (float(v) for v in cyl_xy_traj[ci, idx])
                px, py = (float(v) for v in cyl_xy_traj[ci - 1, idx])
                vx, vy = cx - px, cy - py
                speed = (vx ** 2 + vy ** 2) ** 0.5
                if speed > 1e-6:
                    ux, uy = vx / speed, vy / speed
                    arrow_len = 0.15
                    ax.add_patch(FancyArrowPatch(
                        (cx, cy), (cx + ux * arrow_len, cy + uy * arrow_len),
                        arrowstyle="-|>", mutation_scale=12,
                        color="darkorange", linewidth=1.6, alpha=0.9, zorder=6,
                    ))

        # ── predicted horizon: raw diffusion output vs. projected/safe ──────────
        # cand_traj_xy(_proj) is (H+1, 2) for old npz files, (H+1, 3) for new
        # ones that also captured z. The xy plane only ever needs columns 0/1
        # (always present); the xz plane needs column 2 (z), so it's only
        # drawn when that's actually available -- old files silently skip it.
        chosen_k = None
        if has_horizon:
            si = t - 1  # snap i corresponds to xyz[i+1]
            if 0 <= si < cand_traj_xy.shape[0]:
                chosen_k = int(snap_chosen[si])
                raw_pt  = cand_traj_xy[si, chosen_k]   # (H+1, 2 or 3)
                proj_pt = cand_traj_xy_proj[si]        # (H+1, 2 or 3)
                has_z = raw_pt.shape[-1] >= 3 and proj_pt.shape[-1] >= 3
                if plane == "xy" or has_z:
                    ax.plot(raw_pt[:, h_idx], raw_pt[:, v_idx],
                            color="#5b8fd6", linewidth=1.6, linestyle="--",
                            marker="o", markersize=2.5, markevery=1,
                            alpha=0.85, zorder=4, label="diffusion horizon (raw)")
                    ax.plot(proj_pt[:, h_idx], proj_pt[:, v_idx],
                            color="#d62728", linewidth=1.6, linestyle="-",
                            marker="o", markersize=2.5, markevery=1,
                            alpha=0.95, zorder=5, label="projected horizon (safe)")

        # ── executed trail so far ────────────────────────────────────────────────
        ax.plot(hs_all[:t + 1], vs_all[:t + 1], color="#e87020", linewidth=1.6,
                alpha=0.9, zorder=6, label="executed trail")
        ax.plot(hs_all[0], vs_all[0], "o", markerfacecolor="none",
                markeredgecolor="black", markersize=8, markeredgewidth=1.4,
                zorder=7, label="start")
        ax.plot(hs_all[t], vs_all[t], "o", color="black", markersize=6,
                zorder=8, label="current step")

        # ── goal line ─────────────────────────────────────────────────────────
        if plane == "xy":
            ax.plot([4.0, 4.0], [-1.0, 1.0], color="#00aa00", linewidth=2.2, zorder=5)
        else:
            ax.plot([4.0, 4.0], [0.0, 1.0], color="#00aa00", linewidth=2.2, zorder=5)

        # ── info box ──────────────────────────────────────────────────────────
        info_lines = [
            f"Variant: {VARIANT_LABELS.get(variant, variant)}",
            f"Step: {t}/{T - 1}",
            f"Batch size (K): {K}",
        ]
        if chosen_k is not None:
            info_lines.append(f"Selected candidate: {chosen_k}")
        info_lines.append(f"Tighten margin: {rec.get('tighten', 0.0):.3g} m")
        info_lines.append(f"Projection mode: {rec.get('projection_mode', 'none')}")
        info_lines.append(f"Selection: {rec.get('selection', 'first')}")
        if t == T - 1 and rec.get("success") is not None:
            info_lines.append(f"Outcome: {'success' if rec['success'] else ('fell' if rec['fell'] else 'fail')}")
        ax.text(0.02, 0.97, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=8.5, verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="#f2e2b0", edgecolor="#8a7a4a", alpha=0.9),
                zorder=10)

        ax.set_xlabel(h_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(v_label, fontsize=14, fontweight="bold")
        ax.set_title(f"{VARIANT_LABELS.get(variant, variant)}  –  Step {t}/{T - 1}",
                     fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        # zoom to the current position so the (cm-scale) candidate fan is
        # actually visible; full-corridor context is intentionally sacrificed
        zoom_hw = 0.6
        ax.set_xlim(hs_all[t] - zoom_hw, hs_all[t] + zoom_hw)
        ax.set_ylim(vs_all[t] - zoom_hw, vs_all[t] + zoom_hw)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  fontsize=8.5, framealpha=0.9)
        fig.tight_layout(rect=[0, 0, 0.80, 1])

        out_path = os.path.join(variant_dir, f"step_{t:03d}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[SAVED] {T} step PDFs -> {variant_dir}/")


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
    p.add_argument("--dir", type=str, nargs="+", default=None,
                   help="One or more directories with traj_*.npz files "
                        "(all experiments).")
    p.add_argument("--files",   nargs="*", default=None,
                   help="Explicit .npz file list.")
    p.add_argument("--demo",    action="store_true")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for per-experiment PDFs.")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--plane",   type=str, default="xy", choices=["xy", "xz"],
                   help="Projection plane: 'xy' top-down (default) or "
                        "'xz' side view showing altitude vs corridor progress.")
    p.add_argument("--steps",   action="store_true",
                   help="Also emit one PDF per environment step per variant, "
                        "showing the executed trail plus chosen/unchosen "
                        "candidate rollouts, into <out_dir>/steps_<variant>/.")
    p.add_argument("--combine2x2", action="store_true",
                   help="Instead of one xy-or-xz PDF per encoder, combine the "
                        "two encoders found in --dir into a single 2x2 figure: "
                        "xy row on top, xz row on bottom, non-'vitp' encoder on "
                        "the left and 'vitp' on the right. Requires --dir to "
                        "resolve to exactly two encoders.")
    p.add_argument("--out",     type=str, default=None,
                   help="Output file path for --combine2x2 (default: "
                        "<out_dir>/combined_2x2.pdf).")
    args = p.parse_args()

    # ── collect records grouped by (encoder, latent_dim) ─────────────────────
    grouped = defaultdict(list)   # {(encoder, latent_dim): [rec, ...]}

    if args.demo:
        print("[INFO] Demo mode.")
        grouped = _make_demo_data()

    else:
        if args.dir:
            files = sorted(
                f for d in args.dir
                for f in glob.glob(os.path.join(d, "traj_*.npz"))
            )
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

    out_dir = args.out_dir or (args.dir[0] if args.dir else ".")

    if args.combine2x2:
        if len(grouped) != 2:
            print(f"[ERROR] --combine2x2 requires npz files from exactly 2 "
                  f"encoders in --dir, found {len(grouped)}: "
                  f"{[enc for enc, _ in grouped]}")
            return
        keys = sorted(grouped.keys(), key=lambda k: k[0] == "vitp")
        labels = tuple(ENCODER_DISPLAY.get(enc, enc) for enc, _ in keys)
        out_path = args.out or os.path.join(out_dir, "combined_2x2.pdf")
        print(f"[INFO] Combining {labels[0]} (left) + {labels[1]} (right) "
              f"-> {out_path}")
        plot_experiment_2x2([grouped[keys[0]], grouped[keys[1]]],
                            out_path=out_path, show=not args.no_show,
                            labels=labels)
        return

    # ── one figure per experiment ─────────────────────────────────────────────
    plane_suffix = f"_{args.plane}" if args.plane != "xy" else ""
    for (enc, ldim), records in sorted(grouped.items()):
        fname = f"all_variants_{enc}_L{ldim}{plane_suffix}.pdf"
        out_path = os.path.join(out_dir, fname)
        print(f"[INFO] Plotting {enc} L{ldim}  ({len(records)} variants) "
              f"[plane={args.plane}] -> {fname}")
        plot_experiment(records, out_path=out_path, show=not args.no_show,
                        plane=args.plane)

    # ── per-step PDFs (chosen + unchosen candidates) ──────────────────────────
    if args.steps:
        for (enc, ldim), records in sorted(grouped.items()):
            for rec in records:
                print(f"[INFO] Per-step plots for {rec['variant']} "
                      f"({enc} L{ldim}) [plane={args.plane}]")
                plot_variant_steps(rec, out_dir=out_dir, plane=args.plane)


if __name__ == "__main__":
    main()