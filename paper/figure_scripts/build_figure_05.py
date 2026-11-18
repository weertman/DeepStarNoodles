#!/usr/bin/env python3
"""Build Figure 5: trace-derived kinematics of quiescent -> still-alert transitions.

Run ``prepare_figure_05_measures.py``, ``prepare_figure_05_summary.py``,
``prepare_figure_05_example.py``, ``prepare_figure_05_tank.py`` and
``prepare_figure_05_occupancy.py`` first (traces: ``trace_sources.py``).
Panel D splits each paired effect by the light schedule (``summary.json``
``light_strata``, added 2026-10-17): all boundaries, onsets near a scheduled light
switch and onsets away from one, on a log ratio axis.
Needs NumPy and matplotlib only.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'figure_05_data'
STEM = 'figure_05_q_to_sa_kinematics'
W, H = 7.48, 6.1
INK, MUTED, GRID, LINE = '#20303c', '#5d6d79', '#d8e0e4', '#c9d1d7'
BLUE, TEAL, AMBER, PURPLE = '#35668f', '#237b76', '#93651e', '#6a4c93'
STATE_COLORS = {'quiescent': '#d2a73b', 'SA': '#287ba9', 'moving': '#c0504d'}
STATE_NAMES = {'quiescent': 'Quiescent', 'SA': 'Still alert', 'moving': 'Moving'}
NEAR_SWITCH, AWAY_SWITCH = PURPLE, '#7f8b95'   # D: onsets near a scheduled light switch (emphasis) / away (neutral)
STRATA_OFFSET = .27
GAMMA = 0.7
LABEL_RESOLUTION_S = 60
MEASURES = [('tip_motion', 'Arm-tip motion'), ('arm_deformation', 'Arm deformation'), ('body_speed', 'Body speed'),
            ('support_area', 'Support area'),
            ('fold_asymmetry', 'Fold asymmetry'), ('arm_length_spread', 'Arm-length spread'),
            ('arm_curvature', 'Arm curvature')]


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def running_median(values: np.ndarray, width: int = 15) -> np.ndarray:
    half = width // 2
    return np.array([np.nanmedian(values[max(0, i - half):i + half + 1])
                     if np.isfinite(values[max(0, i - half):i + half + 1]).sum() >= half else np.nan
                     for i in range(len(values))])


def build() -> None:
    measures = json.loads((DATA / 'measures.json').read_text())
    example = json.loads((DATA / 'example.json').read_text())
    summary = json.loads((DATA / 'summary.json').read_text())
    assert digest(DATA / 'example.npz') == example['example_npz_sha256']
    assert digest(DATA / 'summary.npz') == summary['summary_npz_sha256']
    ex = np.load(DATA / 'example.npz')
    tank_meta = json.loads((DATA / 'tank.json').read_text())
    assert digest(DATA / 'tank.npz') == tank_meta['tank_npz_sha256']
    tank = np.load(DATA / 'tank.npz')
    occ_meta = json.loads((DATA / 'occupancy.json').read_text())
    assert digest(DATA / 'occupancy.npz') == occ_meta['occupancy_npz_sha256']
    sm = np.load(DATA / 'summary.npz')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'deepstarnoodles-figure-05-v2'})
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

    def onset_marks(ax):
        ax.axvspan(-LABEL_RESOLUTION_S, 0, color='#eef1f3', lw=0, zorder=0)
        ax.axvline(0, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=1)

    # ------------------------------------------------------------ A: example frames
    heading(.08, .15, 'A', 'Arm traces over 60 s')
    size = 1.55
    for k, (key, label, state) in enumerate((('Q', 'Quiescent', 'quiescent'), ('SA', 'Still alert', 'SA'))):
        x = .08 + k * (size + .12)
        ax = ax_in(x, .38, size, size)
        ax.imshow(0.55 * np.clip(ex[f'rgb_{key}'] / 255.0, 0, 1) ** GAMMA, interpolation='lanczos')
        window = example['windows'][key]
        # Every traced arm at every frame of the 60-s window, body-relative: tight bundles = still arms.
        for ghost in window['arm_ghosts']:
            g = np.asarray(ghost)
            ax.plot(g[:, 0], g[:, 1], color='white', lw=.45, alpha=.16, solid_capstyle='round')
        cx, cy = window['centroid']
        half = 0.62 * 512 / 2
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(color=STATE_COLORS[state], linewidth=1.0)
        text_in(x + size / 2, .38 + size + .11, label, 6.4, STATE_COLORS[state], 'bold', ha='center')

    # ------------------------------------------------------------ B: example time series
    bx, bw = 3.62, 3.78
    heading(bx, .15, 'B', 'One animal through onset')
    lab = ax_in(bx + .35, .38, bw - .35, .1)
    for t, name in zip(ex['label_times'], ex['label_names']):
        lab.add_patch(Rectangle((t, 0), 60, 1, fc=STATE_COLORS.get(str(name), '#bbbbbb'), ec='white', lw=.4))
    lab.set_xlim(-300, 300); lab.set_ylim(0, 1); lab.axis('off')
    text_in(bx + .3, .43, 'Human', 6.0, INK, ha='right')
    rel = ex['relative_s']
    traces = (('tip_motion', 'Tip motion', '(R₀/s)'), ('arm_deformation', 'Deformation', '(R₀/s)'),
              ('fold_asymmetry', 'Fold asymmetry', '(R₀)'))
    for k, (key, label, unit) in enumerate(traces):
        top_in = .64 + k * .45
        ax = ax_in(bx + .35, top_in, bw - .35, .32)
        text_in(bx + .35, top_in - .065, f'{label} {unit}', 6.0, INK)
        onset_marks(ax)
        ax.plot(rel, ex[key], color=LINE, lw=.5, zorder=2)
        ax.plot(rel, running_median(ex[key]), color=INK, lw=1.0, zorder=3)
        ax.set_xlim(-300, 300); ax.set_xticks([-300, -150, 0, 150, 300])
        ax.set_ylim(0, np.nanpercentile(ex[key], 99) * 1.1)
        light_axes(ax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        if k < len(traces) - 1:
            ax.tick_params(labelbottom=False, bottom=False)
        else:
            ax.set_xlabel('Time from onset (s)', fontsize=6.3, labelpad=1.5)
    lx = bx + 2.1   # state key on the heading line: only states present in this example
    present = {str(n) for n in ex['label_names']}
    for name in [n for n in ('quiescent', 'SA', 'moving') if n in present]:
        fig.add_artist(Rectangle((lx / W, 1 - .15 / H - .004), .1 / W, .008, transform=fig.transFigure,
                                 fc=STATE_COLORS[name], ec='none'))
        text_in(lx + .14, .15, STATE_NAMES[name], 6.0, INK)
        lx += .3 + .052 * len(STATE_NAMES[name])

    # ------------------------------------------------------------ C: all boundaries, aligned
    c_top = 2.3
    heading(.08, c_top, 'C', f"All {summary['n_boundaries']} boundaries")
    grid = sm['grid']
    pw, pg = .74, .26
    for k, (key, label) in enumerate(MEASURES):
        x = .42 + k * (pw + pg)
        ax = ax_in(x, c_top + .36, pw, .9)
        onset_marks(ax)
        ax.axhline(1, color=GRID, lw=.6, zorder=1)
        ax.fill_between(grid, sm[f'{key}_q25'], sm[f'{key}_q75'], color=TEAL, alpha=.18, lw=0, zorder=2)
        ax.plot(grid, sm[f'{key}_median'], color=INK, lw=1.0, zorder=3)
        ax.set_xlim(-300, 300); ax.set_xticks([-200, 0, 200], ['−200', '0', '200'])
        lo = np.nanmin(sm[f'{key}_q25']); hi = np.nanmax(sm[f'{key}_q75'])
        pad = (hi - lo) * .08
        ax.set_ylim(lo - pad, hi + pad)
        light_axes(ax)
        ax.set_title(label, fontsize=6.4, color=INK, fontweight='bold', loc='left', pad=3)
        if k == 0:
            ax.set_ylabel('× baseline', fontsize=6.2, labelpad=2)
    text_in(.42 + (len(MEASURES) * pw + (len(MEASURES) - 1) * pg) / 2, c_top + 1.49, 'Time from onset (s)', 6.3, INK, ha='center')

    # ------------------------------------------------------------ D: paired effects, split by the light schedule
    # Per measure: all boundaries (black) and the onsets near a scheduled light switch or away from one (colour).
    # Log ratio axis, because onsets near a switch reach several-fold changes; ticks are labelled in percent.
    d_top = c_top + 1.74
    heading(.08, d_top, 'D', 'Before vs after onset')
    dx, dw, inc_x = 1.3, 2.75, 4.17
    dh = 1.36
    d_ylim = (-.55, len(MEASURES) - .45)
    dax = ax_in(dx, d_top + .3, dw, dh)
    counts = summary['light_context']['counts']
    strata = ((STRATA_OFFSET, 'near_light_switch', NEAR_SWITCH, f"Near a light switch ({counts['near_light_switch']})"),
              (0.0, 'all', INK, f"All ({summary['n_boundaries']})"),
              (-STRATA_OFFSET, 'away_from_light_switch', AWAY_SWITCH,
               f"Away from a switch ({counts['away_from_light_switch']})"))
    rows = list(range(len(MEASURES)))[::-1]
    for row, (key, label) in zip(rows, MEASURES):
        for offset, stratum, color, _ in strata:
            e = summary['light_strata'][key][stratum]['animal']
            if stratum == 'all':
                assert e == summary['effects'][key]['all']
            mid, (lo, hi) = 1 + e['median_change'], (1 + v for v in e['ci95'])
            big = stratum == 'all'
            dax.plot([lo, hi], [row + offset] * 2, color=color, lw=1.0 if big else .8, solid_capstyle='butt', zorder=3)
            dax.plot(mid, row + offset, 'o', ms=3.6 if big else 2.8, color=color, mec='white', mew=.3, zorder=4)
        y_row = d_top + .3 + (d_ylim[1] - row) / (d_ylim[1] - d_ylim[0]) * dh
        text_in(.3, y_row, label, 6.3, INK)
        text_in(inc_x, y_row, f"{100 * summary['effects'][key]['all']['fraction_increased']:.0f}%", 6.3, INK)
    dax.axvline(1, color=MUTED, lw=.6, zorder=1)
    dax.set_xscale('log')
    # Fixed range for the current data; widened (never clipped) if a rerun's intervals fall outside it.
    ends = [1 + v for key, _ in MEASURES for _, s, _, _ in strata for v in summary['light_strata'][key][s]['animal']['ci95']]
    xlim = (min(.76, min(ends) / 1.03), max(7.6, max(ends) * 1.03))
    ticks = [0, 50, 100, 200, 400] + ([800] if xlim[1] > 9.5 else [])
    dax.set_xticks([1 + t / 100 for t in ticks], [f'{t:d}'.replace('-', '−') for t in ticks])
    dax.xaxis.set_minor_locator(plt.NullLocator())
    dax.set_xlim(*xlim)
    dax.set_ylim(*d_ylim)
    dax.set_yticks([])
    dax.set_xlabel('Change after onset (%)', fontsize=6.3, labelpad=1.5)
    light_axes(dax)
    dax.spines['left'].set_visible(False)
    text_in(inc_x, d_top + .15, 'Increased', 6.0, MUTED)
    for k, (_, _, color, name) in enumerate(strata):   # key in the empty lower right of the plot
        ky_ = d_top + .3 + dh - .5 + k * .13
        fig.add_artist(Line2D([(dx + 1.62) / W], [1 - ky_ / H], marker='o', ms=2.8, color=color, ls='none',
                              transform=fig.transFigure))
        text_in(dx + 1.69, ky_, name, 5.8, INK)

    # ------------------------------------------------------------ E: where in the tank (baseline body centre)
    ex_ = 4.82
    heading(ex_, d_top, 'E', 'Position in the tank')
    m = tank['canonical']
    ew = 2.35
    eh = ew * (m[2, 1] - m[0, 1])
    eax = ax_in(ex_ + .2, d_top + .3, ew, eh)
    rim, floor = np.vstack([m[:4], m[:1]]), np.vstack([m[4:8], m[4:5]])
    occ = np.load(DATA / 'occupancy.npz')
    dens = occ['density'] / occ['density'].max()
    occ_cmap = LinearSegmentedColormap.from_list('occ', ['#ffffff', '#cfe3e1', TEAL, '#123f3c'])
    eax.imshow(np.sqrt(dens), cmap=occ_cmap, vmin=0, vmax=1, origin='upper', interpolation='bilinear',
               extent=(occ['x_edges'][0], occ['x_edges'][-1], occ['y_edges'][-1], occ['y_edges'][0]), zorder=0)
    eax.plot(*rim.T, color=MUTED, lw=.7, zorder=1)
    eax.plot(*floor.T, color=MUTED, lw=.5, zorder=1)
    for i in range(4):
        eax.plot(*np.vstack([m[i], m[i + 4]]).T, color=MUTED, lw=.5, zorder=1)
    eax.text(m[4:8, 0].mean(), m[4:8, 1].mean(), 'Floor', fontsize=5.8, color=MUTED, ha='center', va='center',
             zorder=2)
    eax.plot(*tank['xy'].T, marker='o', ms=2.6, mfc=AMBER, mec='white', mew=.4, ls='none', zorder=3)
    eax.set_xlim(m[0, 0], m[1, 0]); eax.set_ylim(m[2, 1], m[0, 1])
    eax.set_aspect('equal'); eax.axis('off')
    # key under the map: occupancy ramp and transition marker
    ky = d_top + .3 + eh + .14
    kax = ax_in(ex_ + .2, ky - .03, .42, .06)
    kax.imshow(np.sqrt(np.linspace(0, 1, 256))[None], cmap=occ_cmap, aspect='auto', extent=(0, 1, 0, 1))
    kax.set_xticks([]); kax.set_yticks([])
    for spine in kax.spines.values():
        spine.set(color=GRID, lw=.4)
    text_in(ex_ + .67, ky, f"Time spent ({occ_meta['n_animals']} animals)", 5.8, INK)
    tx = ex_ + 1.75
    fig.add_artist(Line2D([tx / W], [1 - ky / H], marker='o', ms=2.6, mfc=AMBER, mec='white', mew=.4, ls='none',
                          transform=fig.transFigure))
    text_in(tx + .07, ky, f"Q → SA onset ({summary['n_boundaries']})", 5.8, INK)

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles Figure 5: kinematics of Q->SA transitions', 'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)
    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'traces': measures['traces'],
        'example': {'key': example['key'], 'selection': example['selection'], 'videos': example['videos']},
        'inputs_not_from_the_published_models': {
            'label_boundaries': 'Sleepy Stars human-reviewed state labels (Q to SA onsets)',
            'light_strata': 'experimenter light protocol tables',
            'tank_geometry': 'hand-placed tank markers stored with every recording',
            'occupancy': 'capture-time StarSeg detections archived by the recording loop (*_starseg_frames.jsonl); '
                         'they locate the animal for the Time spent shading in panel E only'},
        'inputs': {name: digest(DATA / name) for name in ('measures.json', 'measures.npz', 'example.json', 'example.npz',
                                                          'summary.json', 'summary.npz', 'tank.json', 'tank.npz',
                                                          'occupancy.json', 'occupancy.npz')},
        'display': {'rgb_gamma': GAMMA, 'example_running_median_s': 15, 'label_resolution_band_s': [-LABEL_RESOLUTION_S, 0],
                    'example_view_fraction_of_crop': 0.62, 'measures': [k for k, _ in MEASURES],
                    'd_strata': ['near_light_switch', 'all', 'away_from_light_switch'], 'd_bootstrap_unit': 'animal',
                    'd_axis': 'log(1 + change), ticks labelled as percent change'},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
