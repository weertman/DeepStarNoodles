#!/usr/bin/env python3
"""Draw docs/polyline_transforms.png: how ``representations`` turns the GRU's arm polylines into measures.

The top row takes one frame's arms through the steps every arm goes through: traced (crop pixels), centred on
the body origin o, scaled by r (the radius of a disc with the star's area; some outputs use R0, its median over a
baseline) and resampled to 12 points evenly spaced along the arm. The bottom row shows what is measured on each
arm, with its output column: length along the arm, curvature (how far the arm's direction turns, divided by its
length), the tip's angle about o (counterclockwise from the right) and shape change (the arm one frame earlier,
shifted and rotated onto it; arm_deformation is the frame's median RMS gap per second). Every value is computed
with the vendored noodle_polylines, as ``representations`` computes it.

Needs the tracked records of one video (``track-videos`` output, .jsonl) and the video itself, plus OpenCV
and matplotlib in the analysis environment that README.md sets up:

    python docs/figures/polyline_transforms.py tracked/flip_i1.jsonl flip_i1.mp4 --frame 26 --arm 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
import numpy as np

from deepstarnoodles_inference._vendor.noodle_polylines import compute_all, geometry
from deepstarnoodles_inference.representations import load_inference

# The DeepStarNoodles paper's figure style: plain ink headings, colour only where it encodes data.
INK, MUTED, LINE, FAINT = "#20303c", "#586976", "#d7dfe3", "#9aa6af"
ARM = "#c4642a"  # the one arm followed through every panel
W, H = 7.5, 4.8  # inches
SIDE, GAP, LEFT = 1.5, 0.36, 0.3  # panel side, gap between panels, left margin (inches)
ROW_A, ROW_B = 0.62, 2.98  # panel tops (inches from the top)
MONO = "DejaVu Sans Mono"


def crop_image(video: Path, frame: int, crop) -> np.ndarray:
    """The 512-pixel crop Hydra and the GRU saw, cut from the video frame."""
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame} of {video}")
    left, top, side = (int(round(value)) for value in crop)
    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    square = padded[top + side:top + 2 * side, left + side:left + 2 * side]
    return cv2.cvtColor(cv2.resize(square, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def kabsch_fit(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``moving`` shifted and rotated (no reflection) onto ``target``, as arm_deformation fits them."""
    a, b = moving - moving.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return a @ (u @ np.diag([1.0, d]) @ vt) + target.mean(0)


def clear_direction(points: np.ndarray, near: float, far: float) -> tuple[float, int]:
    """The angle (radians, y up) farthest from any of ``points`` between ``near`` and ``far`` of the origin, and the
    side (+1 counterclockwise, -1 clockwise) with more room for a label."""
    distance = np.hypot(*points.T)
    angles = np.arctan2(*points[(distance > near) & (distance < far)][:, ::-1].T)

    def clearance(theta):
        return np.min(np.abs(np.angle(np.exp(1j * (angles - theta)))))

    candidates = np.radians(np.arange(0, 360, 2))
    best = float(candidates[int(np.argmax([clearance(c) for c in candidates]))])
    return best, 1 if clearance(best + 0.3) >= clearance(best - 0.3) else -1


def body_coords(points: np.ndarray, origin: np.ndarray, radius: float) -> np.ndarray:
    """Crop pixels to multiples of r about the body origin, y up (geometry.body_path)."""
    return np.column_stack(((points[:, 0] - origin[0]) / radius, (origin[1] - points[:, 1]) / radius))


def example(records: Path, frame: int, arm: int, star: int | None) -> dict:
    stars = load_inference(records)
    c = stars[star if star is not None else sorted(stars)[0]]
    out = compute_all(c, "fig4b")
    position = int(np.flatnonzero(np.asarray(c.frame_index) == frame)[0])
    fo, ao = c.frame_arm_offsets, c.arm_point_offsets

    def arms_of(pos):
        return list(range(fo[pos], fo[pos + 1]))

    def find(pos):
        found = [a for a in arms_of(pos) if c.track_id[a] == arm]
        if not found:
            raise SystemExit(f"arm {arm} is not in frame {int(c.frame_index[pos])}")
        return found[0]

    a, previous = find(position), find(position - 1)
    origin, radius = c.body_origin_model[position], float(out["radius_model"][position])
    base = geometry.resample_base(c.points, ao)  # 12 points per arm, evenly spaced along it
    prev_origin = c.body_origin_model[position - 1]
    # In video pixels both shapes share one radius, as arm_deformation's shapes share R0.
    prev_radius = radius * c.crop[position, 2] / c.crop[position - 1, 2]
    return {
        "crop": c.crop[position], "origin": origin, "radius": radius,
        "arms": [c.points[ao[i]:ao[i + 1]].astype(float) for i in arms_of(position)],
        "highlight": arms_of(position).index(a),
        "points": c.points[ao[a]:ao[a + 1]].astype(float),
        "base": body_coords(base[a], origin, radius),
        "previous": body_coords(base[previous], prev_origin, prev_radius),
        "arc_length": float(out["arc_length"][a]), "norm_length": float(out["norm_length"][a]),
        "curvature": float(out["curvature"][a]), "tip_angle": float(out["tip_angle_deg"][a]),
    }


def build(records: Path, video: Path, frame: int, arm: int, star: int | None, out: Path) -> Path:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.0, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "xtick.major.size": 2.0, "xtick.color": MUTED,
                         "xtick.labelcolor": MUTED, "xtick.labelsize": 5.6})
    ex = example(records, frame, arm, star)
    o, r = ex["origin"], ex["radius"]
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    def text(x, y, s, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("va", "baseline")
        return fig.text(x / W, 1 - y / H, s, **kw)

    def panel(column, top, title, output=None):
        """A square panel, its title above and, for a measure, its output column below."""
        left = LEFT + column * (SIDE + GAP)
        ax = fig.add_axes([left / W, 1 - (top + SIDE) / H, SIDE / W, SIDE / H])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        text(left, top - 0.08, title, fontsize=7.0)
        if output:
            text(left, top + SIDE + 0.17, output, fontsize=6.2, color=MUTED, family=MONO)
        return ax

    def axis_ticks(ax, ticks, labels=None):
        ax.set_xticks(ticks)
        if labels is not None:
            ax.set_xticklabels(labels)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(LINE)

    def origin_mark(ax, xy, colour=INK, halo=None):
        effects = [patheffects.withStroke(linewidth=2.2, foreground=halo)] if halo else None
        ax.plot(*xy, marker="+", ms=7, mew=0.9, color=colour, zorder=6, path_effects=effects)

    def draw_arms(ax, transform, other, width=0.8):
        for i, points in enumerate(ex["arms"]):
            if i != ex["highlight"]:
                ax.plot(*transform(points).T, color=other, lw=width, solid_capstyle="round", zorder=2)
        ax.plot(*transform(ex["arms"][ex["highlight"]]).T, color=ARM, lw=1.5, solid_capstyle="round", zorder=4)

    for y, letter, title in ((0.3, "A", "Transforms"), (ROW_B - 0.36, "B", "Measures")):
        text(LEFT - 0.02, y, letter, fontsize=10.4, fontweight="bold")
        text(LEFT + 0.2, y, title, fontsize=7.4)

    # Traced, centred and scaled show the same view; only the coordinates change.
    extent_px = max(np.abs(np.vstack(ex["arms"]) - o).max() * 1.1, 1.0)

    # Traced: the arms as the GRU wrote them, base to tip, in pixels of the 512-pixel crop.
    ax = panel(0, ROW_A, "Traced")
    ax.imshow(crop_image(video, frame, ex["crop"]), extent=(0, 512, 512, 0))
    draw_arms(ax, lambda p: p, "white", 0.7)
    origin_mark(ax, o, "white", halo=INK)
    label_at, _ = clear_direction(np.vstack([body_coords(p, o, 1.0) for p in ex["arms"]]), 8, 45)
    ax.text(o[0] + 30 * np.cos(label_at), o[1] - 30 * np.sin(label_at), "o", color="white", fontsize=7,
            zorder=6, ha="center", va="center", path_effects=[patheffects.withStroke(linewidth=1.8, foreground=INK)])
    ax.set_xlim(o[0] - extent_px, o[0] + extent_px)
    ax.set_ylim(o[1] + extent_px, o[1] - extent_px)
    axis_ticks(ax, [v for v in range(0, 513, 100) if abs(v - o[0]) < 0.95 * extent_px])

    # Centred on the body origin o (the mouth, or mid-body when it's hidden).
    ax = panel(1, ROW_A, "Centred")
    ax.axhline(0, color=LINE, lw=0.6, zorder=1)
    ax.axvline(0, color=LINE, lw=0.6, zorder=1)
    draw_arms(ax, lambda p: body_coords(p, o, 1.0), FAINT)
    origin_mark(ax, (0, 0))
    ax.set_xlim(-extent_px, extent_px)
    ax.set_ylim(-extent_px, extent_px)
    step = 100 if extent_px > 110 else 50
    axis_ticks(ax, [-step, 0, step])

    # Scaled by r, the radius of a disc with the star's area (dashed).
    extent = extent_px / r
    ax = panel(2, ROW_A, "Scaled")
    ax.axhline(0, color=LINE, lw=0.6, zorder=1)
    ax.axvline(0, color=LINE, lw=0.6, zorder=1)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ls=(0, (2.5, 2)), lw=0.7, color=MUTED, zorder=3))
    free, side = clear_direction(np.vstack([body_coords(p, o, r) for p in ex["arms"]]), 0.25, 1.4)
    ax.plot([0, np.cos(free)], [0, np.sin(free)], color=INK, lw=0.8, zorder=3)
    ax.text(0.6 * np.cos(free) - side * 0.17 * np.sin(free), 0.6 * np.sin(free) + side * 0.17 * np.cos(free), "r",
            color=INK, fontsize=7, ha="center", va="center", style="italic", zorder=6)
    draw_arms(ax, lambda p: body_coords(p, o, r), FAINT)
    origin_mark(ax, (0, 0))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ticks = [-2, -1, 0, 1, 2] if extent > 2.05 else [-1, 0, 1]
    axis_ticks(ax, ticks, [f"{v}r".replace("-", "−") if v else "0" for v in ticks])

    # The zoom shared by the resampled panel and row B: the followed arm, the earlier copy and the origin.
    fitted = kabsch_fit(ex["previous"], ex["base"])
    arm_scaled = body_coords(ex["points"], o, r)
    pts = np.vstack([arm_scaled, fitted, [[0.0, 0.0], [0.62, 0.0]]])
    lo, hi = pts.min(0), pts.max(0)
    half = max(hi - lo) / 2 + 0.38
    centre = (lo + hi) / 2
    zoom = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)

    def zoomed(column, top, title, output=None):
        ax = panel(column, top, title, output)
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
        ax.set_xticks([])
        ax.add_patch(Rectangle((zoom[0], zoom[2]), 2 * half, 2 * half, fill=False, lw=0.6, color=LINE,
                               zorder=0))
        origin_mark(ax, (0, 0))
        return ax

    # Resampled: 12 points evenly spaced along the arm replace the traced ones (grey).
    ax = zoomed(3, ROW_A, "Resampled")
    ax.plot(*arm_scaled.T, color=ARM, lw=0.8, zorder=3)
    ax.plot(*arm_scaled.T, ls="none", marker="o", ms=1.6, color=FAINT, zorder=4)
    ax.plot(*ex["base"].T, ls="none", marker="o", ms=4.2, mfc=ARM, mec="white", mew=0.6, zorder=5)

    # Each step's operation, on the arrow into it.
    for column, operation in enumerate(("− o", "÷ r", f"{len(ex['base'])} pts")):
        x, y = LEFT + (column + 1) * SIDE + column * GAP, ROW_A + SIDE / 2
        fig.patches.append(FancyArrowPatch(((x + 0.07) / W, 1 - y / H), ((x + GAP - 0.07) / W, 1 - y / H),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                           color=MUTED))
        text(x + GAP / 2, y - 0.08, operation, fontsize=6.6, ha="center")

    # Length, summed along the arm: the bar is the arm straightened.
    ax = zoomed(0, ROW_B, "Length", "arc_length")
    ax.plot(*arm_scaled.T, color=ARM, lw=1.5, solid_capstyle="round", zorder=3)
    length = ex["norm_length"]
    y_bar = zoom[3] - 0.28
    x0 = zoom[0] + (2 * half - length) / 2
    ax.plot([x0, x0 + length], [y_bar, y_bar], color=ARM, lw=1.5, solid_capstyle="butt", zorder=3)
    for x in (x0, x0 + length):
        ax.plot([x, x], [y_bar - 0.06, y_bar + 0.06], color=ARM, lw=0.8)
    ax.text(x0 + length / 2, y_bar - 0.12, f"{ex['arc_length']:.0f} px", color=INK, fontsize=6.2, ha="center",
            va="top")

    # Curvature: how far the arm's direction (arrows) turns, divided by its length.
    ax = zoomed(1, ROW_B, "Curvature", "curvature")
    ax.plot(*arm_scaled.T, color=ARM, lw=1.5, solid_capstyle="round", zorder=3)
    shape = ex["base"]
    middle = arm_scaled.mean(0)
    for k in range(0, len(shape) - 1, 2):
        d = (shape[k + 1] - shape[k]) / np.linalg.norm(shape[k + 1] - shape[k])
        normal = np.array([-d[1], d[0]])
        if normal @ (shape[k] - middle) < 0:
            normal = -normal
        start = shape[k] + 0.16 * normal
        ax.annotate("", xy=start + 0.3 * d, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=5.5, shrinkA=0, shrinkB=0),
                    zorder=5)

    # Tip angle about o, counterclockwise from the right.
    ax = zoomed(2, ROW_B, "Tip angle", "tip_angle_deg")
    ax.plot(*arm_scaled.T, color=ARM, lw=1.5, solid_capstyle="round", zorder=3)
    tip = arm_scaled[-1]
    ax.plot([0, tip[0]], [0, tip[1]], color=MUTED, lw=0.7, ls=(0, (2.5, 1.8)), zorder=2)
    ax.plot([0, 0.55], [0, 0], color=MUTED, lw=0.7, zorder=2)
    theta = ex["tip_angle"]
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=theta, lw=0.8, color=INK, zorder=4))
    end = np.radians(theta)
    ax.annotate("", xy=(0.31 * np.cos(end), 0.31 * np.sin(end)),
                xytext=(0.31 * np.cos(end - 0.25), 0.31 * np.sin(end - 0.25)),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=5, shrinkA=0, shrinkB=0))
    mid = np.radians(theta / 2)
    ax.text(0.52 * np.cos(mid), 0.52 * np.sin(mid), f"{theta:.0f}°", color=INK, fontsize=6.2, ha="center",
            va="center")

    # Shape change: the arm one frame earlier (grey), shifted and rotated onto it; the gaps are what changed.
    ax = zoomed(3, ROW_B, "Shape change", "arm_deformation")
    for p, q in zip(fitted, shape):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=0.6, zorder=3)
    ax.plot(*fitted.T, color=FAINT, lw=1.0, marker="o", ms=2.6, zorder=2)
    ax.plot(*shape.T, color=ARM, lw=1.0, marker="o", ms=2.6, zorder=4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, help="one video's tracked records (.jsonl)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=26, help="the video frame shown")
    parser.add_argument("--arm", type=int, default=2, help="the arm track ID followed")
    parser.add_argument("--star", type=int, help="the star track ID (default: the first)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "polyline_transforms.png")
    args = parser.parse_args()
    print(build(args.records, args.video, args.frame, args.arm, args.star, args.out))


if __name__ == "__main__":
    main()
