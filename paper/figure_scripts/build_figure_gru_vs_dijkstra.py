#!/usr/bin/env python3
"""Render the GRU tracer vs Dijkstra figure: how each decoder works, and how they compare."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'deepstarnoodles-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'figure_gru_vs_dijkstra_data'
STEM = 'figure_gru_vs_dijkstra'
W, H = 7.48, 6.02
INK, MUTED, GRID = '#20303c', '#5d6d79', '#d8e0e4'
TRACE, STROKE = '#00e4ff', '#111111'
COST_CMAP, COST_MAX = 'magma_r', 6.0
VIEW_MARGIN_PX = 8
WINDOW_STEPS = (0, 9)            # GRU steps whose 33 x 33 input window is drawn


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build() -> None:
    summary = json.loads((DATA / 'summary.json').read_text())
    mechanism = json.loads((DATA / 'mechanism.json').read_text())
    for name, expected in summary['files'].items():
        assert digest(DATA / name) == expected, f'Changed source: {name}'
    assert digest(DATA / 'mechanism.npz') == mechanism['mechanism_npz_sha256'], 'Changed source: mechanism.npz'
    m = np.load(DATA / 'mechanism.npz')
    speed = json.loads((DATA / 'speed.json').read_text())
    assert all(v['images_with_different_match_counts_from_readout'] == 0 for v in speed['agreement_with_readout'].values())
    example = next(e for e in summary['examples'] if e['image_id'] == mechanism['image_id'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': .6,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.hashsalt': 'deepstarnoodles-figure-gru-vs-dijkstra-v2'})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text_in(x, y, s, size=6.5, color=INK, weight='normal', ha='left', va='center', **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)

    def heading(x, y, letter, title):
        text_in(x, y, letter, 10.5, INK, 'bold')
        text_in(x + .22, y - .005, title, 7.4, INK)

    def steps(x, y, items, width=27):
        for k, item in enumerate(items, 1):
            text_in(x, y, str(k), 6.5, INK, 'bold', va='top')
            body = text_in(x + .13, y, textwrap.fill(item, width, break_on_hyphens=False), 6.1, INK, va='top', linespacing=1.35)
            y += .135 * (body.get_text().count('\n') + 1) + .1

    def image_ax(x, y, size, view):
        ax = ax_in(x, y, size, size)
        ax.set(xlim=view[:2], ylim=view[2:][::-1], xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def light_axes(ax):
        ax.spines[['left', 'bottom']].set_color(GRID)
        ax.tick_params(length=2.2, labelsize=6.0, pad=1.5, colors=MUTED)

    def stroked(ax, xy, color=TRACE, lw=.9, dashed=False, **kw):
        line, = ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle='round',
                        ls=(0, (2.2, 1.4)) if dashed else '-', **kw)
        line.set_path_effects([patheffects.Stroke(linewidth=lw + .9, foreground=STROKE), patheffects.Normal()])
        return line

    def label(ax, x, y, s, ha='left', va='center', size=6.0):
        t = ax.text(x, y, s, fontsize=size, color='white', ha=ha, va=va, zorder=9)
        t.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground=STROKE)])

    image = m['image']
    gray = image.astype(np.float32).mean(2) / 255
    support = m['support'].astype(np.float32) > .5
    ys, xs = np.nonzero(support)
    lo, hi = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])
    side = float((hi - lo).max()) + 2 * VIEW_MARGIN_PX
    cx, cy = (lo + hi) / 2
    view = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
    tip = m['tips'][mechanism['shown_tip'], :2]
    size = 2.3

    # A: Dijkstra. Per-pixel cost for one tip (hand-set weights), and the cheapest path to its paired base.
    heading(.04, .16, 'A', 'Dijkstra decoder: fixed rules, no training')
    ax = image_ax(.12, .42, size, view)
    ax.imshow(gray * .55, cmap='gray', vmin=0, vmax=1)
    cost = np.ma.masked_invalid(m['cost_map'].astype(np.float32))
    shown = ax.imshow(np.clip(cost, 0, COST_MAX), cmap=COST_CMAP, vmin=0, vmax=COST_MAX, interpolation='nearest')
    path = m['shown_dijkstra_path']
    stroked(ax, path, lw=1.1, zorder=5)
    ax.scatter(*m['bases'][:, :2].T, marker='^', s=9, facecolor='white', edgecolor=STROKE, linewidth=.4, zorder=6)
    ax.scatter(*tip, s=16, facecolor='white', edgecolor=STROKE, linewidth=.5, zorder=7)
    label(ax, tip[0] + 5, tip[1] - 7, 'tip')
    label(ax, path[0, 0] - 3, path[0, 1] + 9, 'base', ha='center')
    bar = ax_in(.12 + size * .62, .42 + size + .1, size * .38, .06)
    fig.colorbar(shown, cax=bar, orientation='horizontal', ticks=[]).outline.set_linewidth(.4)
    text_in(.12 + size * .62 - .05, .42 + size + .13, 'cost for this tip:  low', 5.8, MUTED, ha='right')
    text_in(.12 + size + .05, .42 + size + .13, 'high', 5.8, MUTED)
    paired, total = mechanism['dijkstra']['tips_paired_with_a_base'], mechanism['dijkstra']['tips_on_support']
    steps(2.52, .5, [
        'A step costs more off the centerline, against the arm direction or onto a different arm (map). '
        'The weights are set by hand.',
        "Dijkstra's algorithm finds each tip's cheapest path.",
        f'Tips are paired with detected bases (triangles). The {total - paired} of {total} tips without one '
        'run to the lowest-progress pixel (C).'], width=23)

    # B: GRU tracer. One walker's steps, its 33 x 33 input window, and the learned stop decision.
    heading(3.80, .16, 'B', 'GRU tracer: learned, one step at a time')
    ax = image_ax(3.88, .42, size, view)
    ax.imshow(image)
    for k in range(mechanism['gru']['walkers']):
        walk = m[f'gru_walk_{k:02d}']
        if k != mechanism['shown_walk'] and len(walk) > 1:
            ax.plot(walk[:, 0], walk[:, 1], color='white', lw=.5, alpha=.45, zorder=3)
    walk, probs = m['shown_walk_positions'], m['shown_walk_probabilities']
    half = mechanism['gru_patch_px'] / 2
    for s in WINDOW_STEPS:
        ax.add_patch(Rectangle(walk[s] - half, 2 * half, 2 * half, fill=False, ec='white', lw=.7, zorder=6,
                               path_effects=[patheffects.Stroke(linewidth=1.5, foreground=STROKE), patheffects.Normal()]))
    stroked(ax, walk, lw=.7, zorder=5)
    ax.scatter(*walk.T, s=5, facecolor=TRACE, edgecolor=STROKE, linewidth=.35, zorder=7)
    ax.scatter(*tip, s=16, facecolor='white', edgecolor=STROKE, linewidth=.5, zorder=8)
    base_state = mechanism['state_order'].index('TERMINATE_BASE')
    label(ax, walk[0, 0] + half, walk[0, 1] - half - 4, '33 × 33 px input', ha='right', va='bottom', size=5.6)
    label(ax, walk[-1, 0] - 4, walk[-1, 1] + 10, f'stop: base\np = {probs[-1, base_state]:.2f}', ha='center', va='top', size=5.6)
    steps(6.31, .5, [
        'Starts at the same tip peaks.',
        'Each step reads a 33\u00a0×\u00a033\u00a0px patch of the dense fields plus RGB image features.',
        'A GRU remembers the path so far and predicts the next step and whether to stop.',
        'Its weights are learned from 4,062 annotated training images.'], width=20)

    # C: both outputs on the same star.
    heading(.04, 3.32, 'C', 'Output on the same star')
    gt = [np.asarray(a) for a in example['ground_truth']]
    small, x0, y0 = 1.5, .12, 3.80
    for col, key, name in ((0, 'gru', 'GRU tracer'), (1, 'dijkstra', 'Dijkstra')):
        ax = image_ax(x0 + col * (small + .06), y0, small, view)
        ax.imshow(image)
        trace = example[key]
        for j, arm in enumerate(gt):
            if j not in trace['matched_ground_truth']:
                stroked(ax, arm, color='white', dashed=True, zorder=4)
        for arm in trace['predictions']:
            stroked(ax, np.asarray(arm), zorder=5)
        text_in(x0 + col * (small + .06) + small / 2, y0 - .1,
                f"{name} · F1 {example['f1_' + key]:.2f}", 6.3, INK, ha='center')
    key_y = y0 + small + .16
    for kx, dashed, name in ((x0, False, 'decoded arm'), (x0 + 1.0, True, 'missed annotated arm')):
        key = ax_in(kx, key_y - .05, .24, .1, facecolor='#4a5862')
        key.set(xlim=(0, 1), ylim=(-1, 1), xticks=[], yticks=[])
        for spine in key.spines.values():
            spine.set_visible(False)
        stroked(key, np.array([[.12, 0.], [.88, 0.]]), color=TRACE if not dashed else 'white', dashed=dashed)
        text_in(kx + .29, key_y, name, 6.0, INK)

    # D: the predeclared test-set comparison, one point per acquisition group.
    contrast = summary['contrast']
    heading(3.40, 3.32, 'D', 'Accuracy')
    ax = ax_in(3.80, 3.80, 1.5, 1.5)
    light_axes(ax)
    g = np.asarray(summary['group_f1']['full_h128']) * 100
    d = np.asarray(summary['group_f1']['dijkstra']) * 100
    n = np.asarray(summary['group_images'])
    ax.plot([0, 100], [0, 100], color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=1)
    ax.scatter(d, g, s=2 + 2.0 * n, facecolor=INK, edgecolor='white', linewidth=.25, alpha=.5, zorder=3)
    ax.set(xlim=(0, 101), ylim=(0, 101), xticks=range(0, 101, 50), yticks=range(0, 101, 50), aspect='equal')
    ax.set_xlabel('Dijkstra F1 (%)', fontsize=6.3, labelpad=2, color=INK)
    ax.set_ylabel('GRU tracer F1 (%)', fontsize=6.3, labelpad=2, color=INK)
    above = int((g > d).sum())
    low, high = sorted(-np.asarray(contrast['ci95_f1']) * 100)
    ax.text(2, 99, f"GRU +{-contrast['difference_f1'] * 100:.2f} points\n[{low:.2f}, {high:.2f}]\n"
            f"Holm p = {contrast['p_holm']:.3f}", fontsize=5.7, color=INK, va='top', linespacing=1.35)
    text_in(3.80 + 1.5 / 2, 5.78, f'GRU higher in {above} of {len(g)} groups', 5.9, INK, ha='center')

    # E: decoding time per image on the same test fields (Hydra excluded).
    heading(5.42, 3.32, 'E', 'Decoding time per image')
    ax = ax_in(6.02, 3.80, 1.38, 1.5)
    light_axes(ax)
    rows = [('dijkstra', 'Dijkstra\nCPU'), ('gru_reference', 'GRU\nGPU'), ('gru_graph', 'GRU, CUDA\ngraphs, GPU')]
    rng = np.random.default_rng(0)
    medians = {}
    for k, (key, _) in enumerate(rows):
        ms = np.asarray(speed['seconds_per_image'][key]) * 1000
        medians[key] = float(np.median(ms))
        ax.scatter(ms, k + rng.uniform(-.22, .22, len(ms)), s=1.6, facecolor=INK, edgecolor='none', alpha=.35, zorder=2)
        ax.plot([medians[key]] * 2, [k - .32, k + .32], color=INK, lw=1.1, zorder=3)
        ax.text(medians[key], k - .38, f'{medians[key]:.0f} ms', fontsize=5.7, color=INK, ha='center', va='bottom')
    ax.set_xscale('log')
    ax.set(xlim=(5, 5000), ylim=(len(rows) - .45, -.62), yticks=range(len(rows)))
    ax.set_yticklabels([label for _, label in rows], fontsize=5.8, linespacing=1.15)
    ax.tick_params(axis='y', length=0, pad=2, colors=INK)
    ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(['10', '100', '1,000'])
    ax.set_xlabel('ms per image (log scale)', fontsize=6.3, labelpad=2, color=INK)
    ax.spines['left'].set_visible(False)
    totals = {key: float(np.sum(speed['seconds_per_image'][key])) for key, _ in rows}
    text_in(6.02 + 1.38 / 2 - .2, 5.78, f"{len(speed['image_ids'])} images: Dijkstra {totals['dijkstra']:.0f} s, "
            f"GRU {totals['gru_graph']:.0f} s", 5.9, INK, ha='center')

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles: GRU tracer vs Dijkstra decoder', 'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)
    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'prepare_scripts_sha256': {name: digest(HERE / name) for name in (
            'prepare_figure_gru_vs_dijkstra_data.py', 'prepare_figure_gru_vs_dijkstra_mechanism.py',
            'prepare_figure_gru_vs_dijkstra_speed.py')},
        'inputs': {name: digest(DATA / name) for name in ('summary.json', 'example_crops.npz', 'common_arms.npz',
                                                          'mechanism.json', 'mechanism.npz', 'speed.json')},
        'readout_sources': summary['sources'],
        'display': {'example_image': mechanism['image_id'], 'shown_tip': mechanism['shown_tip'],
                    'cost_map': f'ridge + identity terms, clipped at {COST_MAX}', 'window_steps': WINDOW_STEPS,
                    'view_margin_px': VIEW_MARGIN_PX},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
