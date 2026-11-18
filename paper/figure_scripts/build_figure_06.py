#!/usr/bin/env python3
"""Build Figure 6: associating traced arms across video frames.

Run ``prepare_figure_06_example.py``, ``prepare_figure_06_flip.py`` and
``prepare_figure_06_stats.py`` first. Needs NumPy and matplotlib only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'deepstarnoodles-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3-D projection)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parents[2] / 'CORE'))
DATA = HERE / 'figure_06_data'
STEM = 'figure_06_arm_association'
W, H = 7.48, 6.65
INK, MUTED, GRID, LINE = '#20303c', '#5d6d79', '#d8e0e4', '#c9d1d7'
STATE_COLORS = {'quiescent': '#d2a73b', 'SA': '#287ba9', 'moving': '#c0504d'}
POSTURE = {'Right Side Up': '#287ba9', 'Being flipped': '#d28d26', 'Upside Down': '#81549b'}
# Association IDs: categorical, cycled; IDs shorter-lived than MIN_COLOUR_S are grey.
ID_PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#e377c2', '#8c564b',
              '#bcbd22', '#393b79', '#637939', '#ad494a', '#3182bd', '#e6550d']
SHORT = '#b8c0c6'
MIN_COLOUR_S = 60
GAMMA = 0.7


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_colours(frames: list[dict], min_span_s: float) -> dict[int, str]:
    """Colour IDs so that IDs close in time and in angle never share a colour; short-lived IDs grey."""
    span: dict[int, list[float]] = {}
    angles: dict[int, list[float]] = {}
    for r in frames:
        for lab, ang in zip(r['labels'], r['tip_angle_deg']):
            span.setdefault(lab, [r['t'], r['t']])[1] = r['t']
            angles.setdefault(lab, []).append(np.radians(ang))
    mean = {lab: np.angle(np.mean(np.exp(1j * np.asarray(v)))) for lab, v in angles.items()}
    colours = {lab: SHORT for lab in span}
    for lab in sorted(span, key=lambda x: span[x][0]):
        if span[lab][1] - span[lab][0] < min_span_s:
            continue
        used = set()
        for other, c in colours.items():
            if c == SHORT or other == lab:
                continue
            overlap = span[other][0] <= span[lab][1] + 30 and span[lab][0] <= span[other][1] + 30
            if overlap:
                used.add(c)
        free = [c for c in ID_PALETTE if c not in used]
        counts = {c: sum(v == c for v in colours.values()) for c in ID_PALETTE}
        colours[lab] = free[0] if free else min(ID_PALETTE, key=counts.get)   # more than 14 concurrent IDs
    return colours


def kymograph(ax, frames: list[dict], colours: dict[int, str], max_gap: float) -> None:
    """Arm-tip angle about the body centre over time; one line per association ID."""
    tracks: dict[int, list[tuple[float, float]]] = {}
    for r in frames:
        for lab, ang in zip(r['labels'], r['tip_angle_deg']):
            tracks.setdefault(lab, []).append((r['t'], ang))
    for lab, pts in tracks.items():
        pts = np.asarray(pts)
        # Break at time gaps and at the 360 -> 0 wrap only; jumps of an ID between arms stay visible.
        breaks = np.flatnonzero((np.diff(pts[:, 0]) > max_gap) | (np.abs(np.diff(pts[:, 1])) > 180)) + 1
        grey = colours[lab] == SHORT
        for seg in np.split(pts, breaks):
            if len(seg) == 1:
                ax.plot(seg[:, 0], seg[:, 1], '.', ms=1.5, color=colours[lab], mew=0, zorder=2 if grey else 3)
            else:
                ax.plot(seg[:, 0], seg[:, 1], color=colours[lab], lw=.6 if grey else .9, zorder=2 if grey else 3,
                        solid_capstyle='round')
    ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])


def tip_anchored_resample_np(poly):
    """The tracker's tip-anchored resampling (K_TRACK points from the tip), for display."""
    from deepstarnoodles.fused_tracker import tip_anchored_resample
    return tip_anchored_resample(np.asarray(poly, np.float32), 12)


def build() -> None:
    ex = json.loads((DATA / 'example.json').read_text())

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'deepstarnoodles-figure-06'})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text_in(x, y, s, size=6.5, color=INK, weight='normal', ha='left', **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, ha=ha, va='center', **kw)

    def heading(x, y, letter, title):
        text_in(x, y, letter, 10.5, INK, 'bold')
        text_in(x + .22, y - .005, title, 7.4, INK)

    def light_axes(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(GRID)
        ax.tick_params(length=2.2, labelsize=6.0, pad=1.5, colors=MUTED)

    colours = id_colours(ex['frames'], MIN_COLOUR_S)
    flip = json.loads((DATA / 'flip.json').read_text())
    t0 = flip['window_s'][0]
    fr = [{**r, 't': r['t'] - t0} for r in flip['frames']]
    fcol = id_colours(fr, 5)

    # ------------------------------------------------------------ A: one real association step, and the IDs through a flip
    heading(.08, .15, 'A', 'How arms are linked from one frame to the next')
    st6 = json.loads((DATA / 'step.json').read_text())
    gate, cap = st6['tracker']['shape_gate'], st6['tracker']['max_match_cost']
    pair, step = st6['pair'], st6['step']
    COL_T, COL_T1 = '#c2255c', '#2b8a3e'          # frame t, frame t + 0.5 s; overlaps blend
    first_seen = []
    for f_ in st6['stack']:
        for x_ in f_['ids']:
            if x_ not in first_seen:
                first_seen.append(x_)
    for x_ in step['track_ids']:
        if x_ not in first_seen:
            first_seen.append(x_)
    num = {x_: k + 1 for k, x_ in enumerate(first_seen)}          # display numbers 1..n
    idc = {x_: ID_PALETTE[k % len(ID_PALETTE)] for k, x_ in enumerate(first_seen)}
    t_arms = [np.asarray(a_) for a_ in pair['t']['arms']]
    t1_arms = [np.asarray(a_) for a_ in pair['t1']['arms']]
    allp = np.vstack(t_arms + t1_arms)
    lim = np.abs(allp).max() + 12

    def star(x, y, size, groups, labels=None):
        ax = ax_in(x, y, size, size)
        for arms, colour, lw, alpha in groups:
            for poly in arms:
                ax.plot(poly[:, 0], poly[:, 1], color=colour, lw=lw, alpha=alpha, solid_capstyle='round')
        if labels:
            for poly, text, colour in labels:
                tip, prev = poly[-1], poly[-5]
                d = (tip - prev) / max(np.linalg.norm(tip - prev), 1e-6)
                ax.text(*(tip + 20 * d), text, fontsize=5.8, color=colour, ha='center', va='center', fontweight='bold')
        ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim); ax.set_aspect('equal'); ax.axis('off')
        return ax

    top, sz = .5, .78
    for k, (groups, label) in enumerate((([(t_arms, COL_T, 1.8, .8)], 't'),
                                         ([(t1_arms, COL_T1, 1.8, .8)], 't + 0.5 s'),
                                         ([(t_arms, COL_T, 1.8, .6), (t1_arms, COL_T1, 1.8, .6)], 'Overlaid'))):
        x = .08 + k * (sz + .06)
        star(x, top + .25, sz, groups)
        text_in(x + sz / 2, top + .13, label, 6.1, COL_T if k == 0 else COL_T1 if k == 1 else INK, ha='center')

    # matrix: every new arm against every open track, overlaid, with its cost
    cost = np.asarray(step['cost'], float)
    tracks = [np.asarray(p_) for p_ in step['track_polys']]            # tip-anchored, tip first
    chosen = {(i, j) for i, j in step['assignment']}
    row_of = {i: j for i, j in chosen}
    ang = lambda p_: np.degrees(np.arctan2(-p_[1], p_[0])) % 360
    cols = sorted(range(len(tracks)), key=lambda j: ang(tracks[j][0]))
    rows = sorted(range(len(t1_arms)), key=lambda i: cols.index(row_of[i]) if i in row_of else 99)
    n_r, n_c = len(rows), len(cols)
    mx_x, cell = 2.78, .2
    mx_y = top + .25
    half_c = 90
    for r_, i in enumerate(rows):
        for c_, j in enumerate(cols):
            ax = ax_in(mx_x + c_ * cell, mx_y + r_ * cell, cell * .94, cell * .94)
            ax.set_xticks([]); ax.set_yticks([])
            v = cost[i, j]
            edge = idc[step['track_ids'][j]] if (i, j) in chosen else GRID
            for sp in ax.spines.values():
                sp.set(color=edge, lw=1.3 if (i, j) in chosen else .4)
            if not np.isfinite(v):
                ax.text(.5, .5, '×', transform=ax.transAxes, ha='center', va='center', fontsize=5.5, color=LINE)
                continue
            a_pts, _ = tip_anchored_resample_np(t1_arms[i])
            centre = np.vstack([a_pts, tracks[j]]).mean(0)
            ax.plot(tracks[j][:, 0], tracks[j][:, 1], color=COL_T, lw=1.1, alpha=.65, solid_capstyle='round')
            ax.plot(a_pts[:, 0], a_pts[:, 1], color=COL_T1, lw=1.1, alpha=.65, solid_capstyle='round')
            ax.set_xlim(centre[0] - half_c, centre[0] + half_c); ax.set_ylim(centre[1] + half_c, centre[1] - half_c)
            ax.text(.04, .96, f'{v:.0f}', transform=ax.transAxes, ha='left', va='top', fontsize=4.6,
                    color=INK if (i, j) in chosen else MUTED, fontweight='bold' if (i, j) in chosen else 'normal')
    for c_, j in enumerate(cols):
        tid = step['track_ids'][j]
        fig.add_artist(Rectangle(((mx_x + c_ * cell + .03) / W, 1 - (mx_y - .06) / H), (cell * .94 - .06) / W, .04 / H,
                                 transform=fig.transFigure, color=idc[tid], lw=0))
        text_in(mx_x + c_ * cell + cell * .47, mx_y - .14, str(num[tid]), 5.6, INK, ha='center')
    for r_, i in enumerate(rows):
        text_in(mx_x - .05, mx_y + r_ * cell + cell * .47, 'abcdefghij'[r_], 5.8, MUTED, 'bold', ha='right')
    text_in(mx_x + n_c * cell / 2, mx_y - .28, 'Tracks (t)', 6.0, COL_T, ha='center')
    text_in(mx_x - .18, mx_y + n_r * cell / 2, 'Arms (t + 0.5 s)', 6.0, COL_T1, ha='center', rotation=90)

    # the same IDs through the flip (time stack, as in Figure 4B), with smoothed arm-tip trajectories
    sx = mx_x + n_c * cell + .25
    ax3 = ax_in(sx - .15, top - .02, 7.45 - sx + .15, 1.78, projection='3d')
    ax3.set_proj_type('ortho')
    tips: dict[int, list[tuple[float, float, float]]] = {}
    for f_ in st6['stack']:
        for poly, tid in zip(f_['arms'], f_['ids']):
            pl = np.asarray(poly) / 50.0
            ax3.plot(np.full(len(pl), f_['t']), pl[:, 0], -pl[:, 1], color=idc[tid], lw=.55, alpha=.35)
            tips.setdefault(tid, []).append((f_['t'], pl[-1, 0], -pl[-1, 1]))
    for tid, pts in tips.items():
        pts = np.asarray(sorted(pts))
        for seg in np.split(pts, np.flatnonzero(np.diff(pts[:, 0]) > 1.01) + 1):   # break at gaps > 2 frames
            if len(seg) < 2:
                continue
            sm = seg.copy()
            if len(seg) >= 3:   # centred 3-frame running mean of the tip position
                for c_ in (1, 2):
                    sm[1:-1, c_] = (seg[:-2, c_] + seg[1:-1, c_] + seg[2:, c_]) / 3
            ax3.plot(sm[:, 0], sm[:, 1], sm[:, 2], color=idc[tid], lw=1.6, solid_capstyle='round')
    t_lo, t_hi = st6['stack_s']
    ax3.set_xlim(t_lo, t_hi); ax3.set_ylim(-4, 4); ax3.set_zlim(-4, 4)
    ax3.set_box_aspect((2.6, 1, 1), zoom=1.25)
    ax3.view_init(elev=19, azim=-124)
    ax3.set_xticks([368, 372, 376]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.tick_params(labelsize=5.8, pad=-2)
    ax3.set_xlabel('Time (s)', fontsize=6.0, labelpad=-6)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_alpha(0)
        axis._axinfo['grid']['color'] = GRID
        axis._axinfo['grid']['linewidth'] = .3
    text_in(sx + .1, top + .13, 'IDs through the flip', 6.1, INK)

    # ------------------------------------------------------------ B: ten minutes through a Q -> SA onset (1 Hz)
    b_top = 2.42
    heading(.08, b_top, 'B', 'Ten minutes through a still-alert onset')
    labels5 = np.load(HERE / 'figure_05_data' / 'example.npz')
    strip = ax_in(.52, b_top + .32, 4.25, .08)
    for t, name in zip(labels5['label_times'], labels5['label_names']):
        if str(name) in STATE_COLORS:
            strip.add_patch(Rectangle((t, 0), 60, 1, fc=STATE_COLORS[str(name)], ec='white', lw=.4))
    strip.set_xlim(-300, 300); strip.set_ylim(0, 1); strip.axis('off')
    bax = ax_in(.52, b_top + .46, 4.25, 1.16)
    kymograph(bax, ex['frames'], colours, max_gap=3)
    bax.axvline(0, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=1)
    bax.set_xlim(-300, 300); bax.set_xticks([-300, -150, 0, 150, 300])
    light_axes(bax)
    bax.set_ylabel('Arm-tip angle (°)', fontsize=6.3, labelpad=2)
    bax.set_xlabel('Time from still-alert onset (s)', fontsize=6.3, labelpad=1.5)
    for k, (state, label) in enumerate((('quiescent', 'Quiescent'), ('SA', 'Still alert'))):
        xx = .52 + k * .8
        fig.add_artist(Rectangle((xx / W, 1 - (b_top + 2.0) / H - .004), .08 / W, .008, transform=fig.transFigure,
                                 fc=STATE_COLORS[state], ec='none'))
        text_in(xx + .11, b_top + 2.0, label, 5.8, INK)

    # ------------------------------------------------------------ C: through a flip (2 Hz)
    cx = 5.08
    heading(cx - .2, b_top, 'C', 'Through a flip')
    strip = ax_in(cx + .3, b_top + .32, 2.0, .08)
    for iv in flip['human_intervals']:
        strip.add_patch(Rectangle((iv['start_s'] - t0, 0), iv['stop_s'] - iv['start_s'], 1, color=POSTURE[iv['state']], lw=0))
    strip.set_xlim(0, flip['window_s'][1] - t0); strip.set_ylim(0, 1); strip.axis('off')
    cax2 = ax_in(cx + .3, b_top + .46, 2.0, 1.16)
    kymograph(cax2, fr, fcol, max_gap=1.6)

    cax2.set_xlim(0, flip['window_s'][1] - t0); cax2.set_xticks([0, 20, 40])
    light_axes(cax2)
    cax2.set_xlabel('Time (s)', fontsize=6.3, labelpad=1.5)
    for k, (state, label) in enumerate((('Right Side Up', 'Right side up'), ('Being flipped', 'Being flipped'),
                                        ('Upside Down', 'Upside down'))):
        yy = b_top + 1.98 + k * .0
        xx = cx + .3 + k * .72
        fig.add_artist(Rectangle((xx / W, 1 - (b_top + 2.0) / H - .004), .08 / W, .008, transform=fig.transFigure,
                                 fc=POSTURE[state], ec='none'))
        text_in(xx + .11, b_top + 2.0, label, 5.8, INK)

    # ------------------------------------------------------------ D: self-consistency over all recordings
    st = json.loads((DATA / 'stats.json').read_text())
    assert digest(DATA / 'stats.npz') == st['stats_npz_sha256']
    summ = st['summary']
    d_top = 4.75
    heading(.08, d_top, 'D', 'Self-consistency across all recordings')
    corpora = (('sleepy', 'Sleepy Stars (1 Hz)', '#20303c'), ('flipping', 'Flipping (2 Hz)', '#7d8b95'))
    # D1: how long IDs last, arm-weighted
    d1 = ax_in(.52, d_top + .32, 1.3, 1.3)
    for key, label, colour in corpora:
        surv = summ[key]['fused']['fraction_arms_in_ids_lasting_ge']
        t = np.array([float(k) for k in surv]); v = np.array(list(surv.values()))
        d1.plot(t, 100 * v, '-o', color=colour, ms=2.2, lw=1.0, label=label)
    d1.set_xscale('log'); d1.set_xticks([1, 10, 100, 600], ['1', '10', '100', '600'])
    d1.minorticks_off()
    d1.set_ylim(0, 102); d1.set_yticks([0, 50, 100])
    light_axes(d1)
    d1.set_xlabel('ID lasts at least (s)', fontsize=6.3, labelpad=1.5)
    d1.set_ylabel('Traced arms (%)', fontsize=6.3, labelpad=2)
    d1.legend(frameon=False, fontsize=5.8, loc='lower left', handlelength=1.2, borderaxespad=.1)

    # D2: frame-to-frame consistency by human-annotated state
    rows = [('sleepy', 'quiescent', 'Quiescent', STATE_COLORS['quiescent']),
            ('sleepy', 'SA', 'Still alert', STATE_COLORS['SA']),
            ('sleepy', 'moving', 'Moving', STATE_COLORS['moving']),
            ('flipping', 'Right Side Up', 'Right side up', POSTURE['Right Side Up']),
            ('flipping', 'Being flipped', 'Being flipped', POSTURE['Being flipped']),
            ('flipping', 'Upside Down', 'Upside down', POSTURE['Upside Down'])]
    ypos = np.array([0, 1, 2, 3.5, 4.5, 5.5])
    cols = (('continuation', 'Continued (%)', (85, 100), [85, 90, 95, 100], 100),
            ('forward_reverse_agreement', 'Same in reverse (%)', (80, 100), [80, 90, 100], 100),
            ('jumps_per_1000_links', 'Jumps per 1,000', (.6, 200), [1, 10, 100], 1))
    tx, tw, tg = 3.0, .74, .26
    for c_i, (metric, label, lim, ticks, scale) in enumerate(cols):
        ax = ax_in(tx + c_i * (tw + tg), d_top + .32, tw, 1.3)
        for (corpus, state, _, colour), y in zip(rows, ypos):
            new = scale * summ[corpus]['fused']['by_state'][state][metric]
            old = scale * summ[corpus]['no_cap']['by_state'][state][metric]
            ax.plot([old, new], [y, y], color=colour, lw=.7, alpha=.6, zorder=2)
            ax.plot(old, y, 'o', mfc='white', mec=colour, mew=.8, ms=3.6, zorder=3)
            ax.plot(new, y, 'o', color=colour, ms=3.8, zorder=4)
        ax.set_ylim(6.1, -.6); ax.set_yticks([])
        if metric == 'jumps_per_1000_links':
            ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xlim(*lim); ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
        light_axes(ax); ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=GRID, lw=.4, zorder=0)
        ax.set_xlabel(label, fontsize=6.0, labelpad=1.5)
    for (corpus, _, label, _), y in zip(rows, ypos):
        y_in = d_top + .32 + 1.3 * (y + .6) / 6.7
        text_in(tx - .06, y_in, label, 5.9, INK, ha='right')
    text_in(tx - .06, d_top + .2, 'Sleepy Stars', 5.9, MUTED, ha='right')
    for k, (label, filled) in enumerate((('With match-cost cap', True), ('Without cap', False))):
        xx = tx + .05 + k * 1.15
        fig.add_artist(Line2D([xx / W], [1 - (d_top + .17) / H], marker='o', ms=3.6, ls='none', transform=fig.transFigure,
                              mfc=MUTED if filled else 'white', mec=MUTED, mew=.8))
        text_in(xx + .07, d_top + .17, label, 5.9, INK)
    text_in(tx - .06, d_top + .32 + 1.3 * (3.5 + .6) / 6.7 - .14, 'Flipping', 5.9, MUTED, ha='right')

    # D3: agreement with full-rate links when frames are skipped
    d3 = ax_in(6.12, d_top + .32, 1.22, 1.3)
    for key, label, colour in corpora:
        fps = 1.0 if key == 'sleepy' else 2.0
        g = np.array([1] + [int(k) for k in summ[key]['stride']])
        prec = np.array([1.0] + [v['precision'] for v in summ[key]['stride'].values()])
        d3.plot(g / fps, 100 * prec, '-o', color=colour, ms=2.2, lw=1.0)
    d3.set_xscale('log', base=2); d3.minorticks_off()
    d3.set_xticks([1, 4, 16], ['1', '4', '16'])
    d3.set_ylim(60, 101); d3.set_yticks([60, 80, 100])
    light_axes(d3)
    d3.set_xlabel('Time between frames (s)', fontsize=6.3, labelpad=1.5)
    d3.set_ylabel('Links kept (%)', fontsize=6.3, labelpad=2)

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles Figure 6: arm association across frames', 'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)
    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'traces': st['traces'],
        'examples': {'A_and_C': {'clip': flip['clip'], 'video': flip['video']},
                     'B': {'boundary': ex['boundary_key'], 'animal': ex['individual_id']}},
        'inputs_not_from_the_published_models': {
            'state_labels': 'human annotations: Sleepy Stars per-minute ethograms and the flipping clips\' posture labels'},
        'inputs': {name: digest(DATA / name) for name in ('step.json', 'example.json', 'flip.json', 'stats.json', 'stats.npz')},
        'display': {'min_colour_lifetime_s': MIN_COLOUR_S, 'rgb_gamma': GAMMA},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
