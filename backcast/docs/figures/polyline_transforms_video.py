#!/usr/bin/env python3
"""Animate docs/polyline_transforms.png as a short video, docs/polyline_transforms.mp4 (1920x1080, 30 fps).

The same arm as the figure goes through the same steps: traced, centred on the body origin o (the coordinate
origin slides onto the body), scaled by r (the ruler changes to multiples of r) and resampled to 12 points. Then
come the measures: the arm straightens to show its length, arrows count up how far it turns, the tip angle sweeps
open, and the arm one frame earlier is shifted and rotated onto it. Every value is computed with the vendored
noodle_polylines, as in the figure.

Needs ffmpeg with libx264 on PATH, plus what polyline_transforms.py needs:

    python docs/figures/polyline_transforms_video.py tracked/flip_i1.jsonl flip_i1.mp4 --frame 26 --arm 2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Arc, Circle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyline_transforms as still  # noqa: E402  (the figure: data loading and shared helpers)

INK, MUTED, LINE, FAINT, ARM = still.INK, still.MUTED, still.LINE, still.FAINT, still.ARM
WIDTH, HEIGHT, FPS = 1920, 1080, 30
STAGE = (860, 60, 900)  # left, top, side of the square stage, in pixels
TEXT_X = 150
# name, second line, whether the second line is an output column, start, end (seconds)
STEPS = [("Traced", "polyline_crop", True, 0.0, 2.0),
         ("Centred", "− o", False, 2.0, 4.0),
         ("Scaled", "÷ r", False, 4.0, 6.0),
         ("Resampled", "12 points", False, 6.0, 8.0),
         ("Length", "arc_length", True, 8.0, 9.8),
         ("Curvature", "curvature", True, 9.8, 11.6),
         ("Tip angle", "tip_angle_deg", True, 11.6, 13.4),
         ("Shape change", "arm_deformation", True, 13.4, 15.5)]
DURATION = STEPS[-1][4]


def ramp(t: float, start: float, end: float) -> float:
    """0 before ``start``, 1 after ``end``, eased in between."""
    x = min(max((t - start) / (end - start), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def linear(t: float, start: float, end: float) -> float:
    return min(max((t - start) / (end - start), 0.0), 1.0)


def mix(a: str, b: str, amount: float) -> tuple:
    """Colour ``a`` moved ``amount`` of the way to ``b``."""
    ca, cb = np.array(matplotlib.colors.to_rgb(a)), np.array(matplotlib.colors.to_rgb(b))
    return tuple(ca + (cb - ca) * amount)


def partial(points: np.ndarray, fraction: float) -> np.ndarray:
    """The first ``fraction`` of a polyline, by length."""
    if fraction >= 1:
        return points
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    stop = fraction * cum[-1]
    keep = cum < stop
    end = np.array([np.interp(stop, cum, points[:, 0]), np.interp(stop, cum, points[:, 1])])
    return np.vstack([points[keep], end])


def rotation(angle: float) -> np.ndarray:
    """Counterclockwise rotation for row vectors (``points @ rotation(a)``)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


class Scene:
    """Everything in world coordinates: crop pixels with y up, so the photo and every later view line up."""

    def __init__(self, records: Path, video: Path, frame: int, arm: int, star: int | None):
        ex = still.example(records, frame, arm, star)
        self.image = still.crop_image(video, frame, ex["crop"])
        self.r = r = ex["radius"]
        self.o = np.array([ex["origin"][0], -ex["origin"][1]])
        self.arms = [np.column_stack((p[:, 0], -p[:, 1])) for p in ex["arms"]]
        self.highlight = ex["highlight"]
        self.arm = self.arms[self.highlight]
        self.base = self.o + r * ex["base"]           # the 12 resampled points
        self.previous = self.o + r * ex["previous"]   # the same arm one frame earlier, in its own body frame
        self.arc_length, self.tip_angle = ex["arc_length"], ex["tip_angle"]
        self.turn = np.degrees(ex["curvature"] * ex["arc_length"])  # total turning, as curvature defines it

        # The shift and rotation that fit the earlier arm onto this one (what arm_deformation measures after).
        self.c_prev, self.c_now = self.previous.mean(0), self.base.mean(0)
        u, _, vt = np.linalg.svd((self.previous - self.c_prev).T @ (self.base - self.c_now))
        fit = u @ np.diag([1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        self.fit_angle = float(np.arctan2(fit[0, 1], fit[0, 0]))

        # The arm resampled finely by length, to straighten it without changing its length.
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(self.arm, axis=0).T))]
        s = np.linspace(0, cum[-1], 161)
        self.dense = np.column_stack((np.interp(s, cum, self.arm[:, 0]), np.interp(s, cum, self.arm[:, 1])))
        self.step = cum[-1] / 160
        self.angles = np.unwrap(np.arctan2(*np.diff(self.dense, axis=0)[:, ::-1].T))
        chord = self.dense[-1] - self.dense[0]
        target = np.arctan2(chord[1], chord[0])
        self.target = target + 2 * np.pi * np.round((self.angles.mean() - target) / (2 * np.pi))
        turning = np.r_[0, np.cumsum(np.abs(np.diff(self.angles)))]
        self.turning = turning * (self.turn / turning[-1]) if turning[-1] > 0 else turning

        # Cameras: the whole crop, the star, and the followed arm (with room for its straightened copy).
        self.extent = np.abs(np.vstack(self.arms) - self.o).max() * 1.12
        near = np.vstack([self.arm, self.unbent(1.0), self.base, self.previous, self.o, self.o + [0.62 * r, 0]])
        lo, hi = near.min(0), near.max(0)
        self.cameras = {"crop": (np.array([256.0, -256.0]), 290.0), "star": (self.o, self.extent),
                        "arm": ((lo + hi) / 2, max(hi - lo) / 2 + 0.4 * r)}
        label_at, _ = still.clear_direction(np.vstack([p - self.o for p in self.arms]), 8, 45)
        self.o_label = self.o + 30 * np.array([np.cos(label_at), np.sin(label_at)])
        free, side = still.clear_direction(np.vstack([(p - self.o) / r for p in self.arms]), 0.25, 1.4)
        self.r_dir = np.array([np.cos(free), np.sin(free)])
        self.r_label = self.o + r * (0.6 * self.r_dir + side * 0.18 * np.array([-self.r_dir[1], self.r_dir[0]]))

    def unbent(self, amount: float) -> np.ndarray:
        """The arm with each segment's direction turned ``amount`` of the way to straight; its length is kept."""
        angles = self.angles * (1 - amount) + self.target * amount
        steps = self.step * np.column_stack((np.cos(angles), np.sin(angles)))
        return np.vstack([self.dense[0], self.dense[0] + np.cumsum(steps, axis=0)])

    def camera(self, t: float) -> tuple[np.ndarray, float]:
        def blend(a, b, amount):
            (ca, ha), (cb, hb) = self.cameras[a], self.cameras[b]
            return ca + (cb - ca) * amount, ha ** (1 - amount) * hb ** amount

        if t < 6.0:
            return blend("crop", "star", ramp(t, 2.25, 3.3))
        return blend("star", "arm", ramp(t, 6.0, 6.8))


class Renderer:
    def __init__(self, scene: Scene):
        self.s = scene
        plt.rcParams.update({"font.family": "DejaVu Sans"})
        self.fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
        self.fig.patch.set_facecolor("white")
        left, top, side = STAGE
        self.ax = self.fig.add_axes([left / WIDTH, 1 - (top + side) / HEIGHT, side / WIDTH, side / HEIGHT])
        self.overlay = []  # figure-level artists of the current frame

    def px(self, x, y):
        """Figure fraction of a pixel position (from the top left)."""
        return x / WIDTH, 1 - y / HEIGHT

    def text(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        self.overlay.append(self.fig.text(*self.px(x, y), s, **kw))

    def line(self, xs, ys, **kw):
        """A line in pixel positions (from the top left), over the whole figure."""
        self.overlay.append(self.fig.add_artist(matplotlib.lines.Line2D(
            [x / WIDTH for x in xs], [1 - y / HEIGHT for y in ys], transform=self.fig.transFigure, **kw)))

    def frame(self, t: float) -> np.ndarray:
        s, ax, r = self.s, self.ax, self.s.r
        for artist in self.overlay:
            artist.remove()
        self.overlay.clear()
        ax.cla()
        ax.set_axis_off()
        centre, half = s.camera(t)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        halo = [patheffects.withStroke(linewidth=4, foreground="white")]

        # The photo, and the other arms: white over the photo, grey once it has gone.
        photo = 1 - ramp(t, 2.0, 2.6)
        if photo > 0:
            ax.imshow(s.image, extent=(0, 512, -512, 0), alpha=photo, zorder=0)
        grown = ramp(t, 0.15, 1.2)
        others = (1 - ramp(t, 6.0, 6.5))
        if others > 0:
            colour = mix(FAINT, "white", photo)
            for i, arm in enumerate(s.arms):
                if i != s.highlight:
                    ax.plot(*partial(arm, grown).T, color=colour, lw=2.4, alpha=others, solid_capstyle="round",
                            zorder=2)

        # Crosshair through the coordinate origin, which slides from the crop's corner onto the body origin,
        # and a ruler under the stage, in crop pixels and then in multiples of r.
        axes_alpha = ramp(t, 1.3, 1.6) * (1 - ramp(t, 6.0, 6.4))
        origin = s.o * ramp(t, 2.3, 3.3)
        if axes_alpha > 0:
            ax.axhline(origin[1], color=LINE, lw=1.6, alpha=axes_alpha, zorder=1)
            ax.axvline(origin[0], color=LINE, lw=1.6, alpha=axes_alpha, zorder=1)
            left, top, side = STAGE
            y = top + side + 22
            self.line([left, left + side], [y, y], color=LINE, lw=1.6, alpha=axes_alpha)
            # Pixels fade out before multiples of r fade in.
            for scale, labels, alpha in ((100.0, lambda k: f"{k * 100:g}", 1 - ramp(t, 4.85, 5.15)),
                                         (r, lambda k: f"{k}r" if k else "0", ramp(t, 5.15, 5.45))):
                if alpha <= 0:
                    continue
                for k in range(-6, 7):
                    x_world = origin[0] + k * scale
                    if abs(x_world - centre[0]) > 0.97 * half:
                        continue
                    x = left + (x_world - centre[0] + half) / (2 * half) * side
                    self.line([x, x], [y, y + 10], color=MUTED, lw=1.4, alpha=axes_alpha * alpha)
                    self.text(x, y + 40, labels(k).replace("-", "−"), fontsize=20, color=MUTED, ha="center",
                              alpha=axes_alpha * alpha)

        # Scaled: a disc with the star's area grows from o; its radius is r.
        if 4.0 <= t < 6.4:
            fade = 1 - ramp(t, 6.0, 6.4)
            grow = ramp(t, 4.0, 4.5)
            fill = grow * (1 - ramp(t, 4.6, 5.0))
            if fill > 0:
                ax.add_patch(Circle(s.o, r * grow, color=FAINT, alpha=0.35 * fill, lw=0, zorder=1))
            outline = ramp(t, 4.5, 4.9) * fade
            if outline > 0:
                ax.add_patch(Circle(s.o, r, fill=False, ls=(0, (4, 3)), lw=2.0, color=MUTED, alpha=outline, zorder=3))
                ax.plot(*np.array([s.o, s.o + r * s.r_dir]).T, color=INK, lw=2.2, alpha=outline, zorder=3)
                ax.text(*s.r_label, "r", fontsize=26, style="italic", color=INK, ha="center", va="center",
                        alpha=outline, zorder=6, path_effects=halo)

        # The followed arm: traced, then straightened for its length, then as 12 points for shape change.
        straight = ramp(t, 8.15, 8.85) * (1 - ramp(t, 9.35, 9.75))
        points_view = ramp(t, 13.4, 13.65)
        arm = s.unbent(straight) if straight > 0 else partial(s.arm, grown)
        if points_view < 1:
            ax.plot(*arm.T, color=ARM, lw=4.2, solid_capstyle="round", alpha=1 - points_view, zorder=4)
        if straight > 0.97:
            label = ramp(t, 8.8, 9.0) * (1 - ramp(t, 9.25, 9.35))
            a, b = arm[0], arm[-1]
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([-d[1], d[0]])
            n = n if n[1] > 0 else -n
            for end in (a, b):
                ax.plot(*np.array([end - 0.08 * r * n, end + 0.08 * r * n]).T, color=ARM, lw=2.2, alpha=label,
                        zorder=4)
            ax.text(*((a + b) / 2 + 0.3 * r * n), f"{s.arc_length:.0f} px", fontsize=24, color=INK, ha="center",
                    va="center", alpha=label, zorder=6, path_effects=halo)

        # Origin mark, and its name on the photo and after.
        ax.plot(*s.o, marker="+", ms=24, mew=2.6, color=INK, zorder=7, alpha=ramp(t, 1.2, 1.5),
                path_effects=[patheffects.withStroke(linewidth=6, foreground="white")])
        name = ramp(t, 1.3, 1.6) * (1 - ramp(t, 4.0, 4.3))  # o's name gives way to r's
        if name > 0:
            ax.text(*s.o_label, "o", fontsize=24, color=INK, ha="center", va="center", alpha=name, zorder=7,
                    path_effects=[patheffects.withStroke(linewidth=5, foreground="white")])

        # Resampled: the traced points, then 12 evenly spaced along the arm.
        traced = ramp(t, 6.5, 6.8) * (1 - ramp(t, 7.6, 8.0))
        if traced > 0:
            ax.plot(*s.arm.T, ls="none", marker="o", ms=5, color=MUTED, alpha=traced, zorder=5)
        shown = 1 - ramp(t, 8.0, 8.2) if t < 13 else points_view
        for k, point in enumerate(s.base):
            pop = ramp(t, 6.9 + 0.055 * k, 7.02 + 0.055 * k) if t < 13 else 1.0
            if pop * shown > 0:
                ax.plot(*point, marker="o", ms=15 * pop, mfc=ARM, mec="white", mew=1.8, alpha=shown, zorder=6)

        # Curvature: arrows along the arm and how far its direction has turned so far.
        if 9.8 <= t < 11.6:
            fade = 1 - ramp(t, 11.4, 11.6)
            progress = linear(t, 9.95, 10.95)
            middle = s.arm.mean(0)
            for k in range(0, len(s.base) - 1, 2):
                if progress < (k + 1) / (len(s.base) - 1):
                    break
                d = (s.base[k + 1] - s.base[k]) / np.linalg.norm(s.base[k + 1] - s.base[k])
                n = np.array([-d[1], d[0]])
                n = n if n @ (s.base[k] - middle) > 0 else -n
                start = s.base[k] + 0.17 * r * n
                ax.annotate("", xy=start + 0.3 * r * d, xytext=start, zorder=5,
                            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK, alpha=fade, mutation_scale=22,
                                            shrinkA=0, shrinkB=0))
            turned = np.interp(progress * (len(s.turning) - 1), np.arange(len(s.turning)), s.turning)
            ax.text(s.arm[:, 0].min(), s.arm[:, 1].max() + 0.1 * r, f"turns {turned:.0f}°", fontsize=28,
                    color=INK, va="bottom", alpha=fade * ramp(t, 9.85, 10.0))

        # Tip angle: from the right, counterclockwise to the tip's direction.
        if 11.6 <= t < 13.5:
            fade = 1 - ramp(t, 13.3, 13.5)
            tip = s.arm[-1]
            reach = ramp(t, 11.65, 11.95)
            ax.plot(*np.array([s.o, s.o + (tip - s.o) * reach]).T, color=MUTED, lw=2.0, ls=(0, (4, 3)), alpha=fade,
                    zorder=3)
            ax.plot(*np.array([s.o, s.o + [0.55 * r * ramp(t, 11.65, 11.85), 0]]).T, color=MUTED, lw=2.0,
                    alpha=fade, zorder=3)
            theta = s.tip_angle * ramp(t, 11.95, 12.75)
            if theta > 1:
                ax.add_patch(Arc(s.o, 0.62 * r, 0.62 * r, theta1=0, theta2=theta, lw=2.4, color=INK, alpha=fade,
                                 zorder=5))
                end = np.radians(theta)
                ax.annotate("", xy=s.o + 0.31 * r * np.array([np.cos(end), np.sin(end)]),
                            xytext=s.o + 0.31 * r * np.array([np.cos(end - 0.2), np.sin(end - 0.2)]), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK, alpha=fade, mutation_scale=20,
                                            shrinkA=0, shrinkB=0))
                mid = np.radians(s.tip_angle / 2)
                ax.text(*(s.o + 0.6 * r * np.array([np.cos(mid), np.sin(mid)])), f"{theta:.0f}°", fontsize=26,
                        color=INK, ha="center", va="center", alpha=fade, zorder=6, path_effects=halo)

        # Shape change: the arm one frame earlier is shifted and rotated onto this one; the gaps are the change.
        if t >= 13.4:
            move = ramp(t, 13.85, 14.65)
            earlier = (s.previous - s.c_prev) @ rotation(s.fit_angle * move) + s.c_prev + (s.c_now - s.c_prev) * move
            ax.plot(*earlier.T, color=FAINT, lw=2.6, marker="o", ms=11, alpha=ramp(t, 13.5, 13.75), zorder=3)
            ax.plot(*s.base.T, color=ARM, lw=2.6, alpha=points_view, zorder=5)
            gaps = ramp(t, 14.65, 14.9)
            if gaps > 0:
                for p, q in zip(earlier, s.base):
                    ax.plot(*np.array([p, q]).T, color=INK, lw=1.8, alpha=gaps, zorder=4)

        # The step's name and its operation or output column, and where we are.
        for i, (title, second, column, start, end) in enumerate(STEPS):
            alpha = ramp(t, start, start + 0.2) if i else 1.0  # one step's words go before the next come
            if i < len(STEPS) - 1:
                alpha *= 1 - ramp(t, end - 0.2, end)
            if alpha <= 0:
                continue
            self.text(TEXT_X, 500, title, fontsize=46, alpha=alpha)
            self.text(TEXT_X, 578, second, fontsize=30 if column else 38, color=MUTED, alpha=alpha,
                      family=still.MONO if column else "DejaVu Sans")
        current = max(i for i, step in enumerate(STEPS) if t >= step[3])
        for i in range(len(STEPS)):
            x = TEXT_X + 6 + i * 34 + (26 if i >= 4 else 0)
            colour, size = (INK, 9) if i == current else ((MUTED, 6) if i < current else (LINE, 6))
            self.line([x], [680], marker="o", ms=size, color=colour)

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]


def render(records: Path, video: Path, frame: int, arm: int, star: int | None, out: Path) -> Path:
    """An MP4, or a looping GIF at half size and half the frame rate when ``out`` ends in .gif."""
    renderer = Renderer(Scene(records, video, frame, arm, star))
    out.parent.mkdir(parents=True, exist_ok=True)
    gif = out.suffix.lower() == ".gif"
    fps = FPS // 2 if gif else FPS
    if gif:  # one palette for the whole clip; ordered dither keeps the file small
        encode = ["-vf", f"scale={WIDTH // 2}:-1:flags=lanczos,split[a][b];[a]palettegen[p];"
                         "[b][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0"]
    else:
        encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart"]
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-", *encode, str(out)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for n in range(int(round(DURATION * fps))):
            encoder.stdin.write(renderer.frame(n / fps).tobytes())
        encoder.stdin.close()
    if encoder.returncode:
        raise SystemExit("ffmpeg failed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, help="one video's tracked records (.jsonl)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=26, help="the video frame shown")
    parser.add_argument("--arm", type=int, default=2, help="the arm track ID followed")
    parser.add_argument("--star", type=int, help="the star track ID (default: the first)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "polyline_transforms.mp4")
    parser.add_argument("--still", type=float, nargs="*", help="write PNGs of these times (s) instead of the video")
    args = parser.parse_args()
    if args.still:
        renderer = Renderer(Scene(args.records, args.video, args.frame, args.arm, args.star))
        for t in args.still:
            path = args.out.with_name(f"{args.out.stem}_{t:05.2f}s.png")
            plt.imsave(path, renderer.frame(t))
            print(path)
        return
    print(render(args.records, args.video, args.frame, args.arm, args.star, args.out))


if __name__ == "__main__":
    main()
