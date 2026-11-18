#!/usr/bin/env python3
"""Build the inference-throughput figure.

Run ``prepare_figure_inference.py`` first.
A: every timed archive run of the agent-run optimization rounds on one RTX 5090,
   by outcome, with the line following the configuration selected after each round.
B: sustained streamed annotation by GPU type with the deployed configuration.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE / 'figure_inference_data'
STEM = 'figure_inference_throughput'
W, H = 7.48, 3.12
INK, MUTED, GRID = '#20303c', '#5d6d79', '#d8e0e4'
FAINT = '#9aa6af'
BAR = '#a3b0b9'
MEMORY_Y = 2.3   # strip for runs stopped at the memory guard (they have no rate)
# Marker per outcome: (marker, size, face, edge, edge width, z-order, legend label)
STYLE = {
    'reference': ('_', 4.5, FAINT, FAINT, .9, 2, 'Original pipeline'),
    'baseline': ('o', 2.0, FAINT, FAINT, .4, 2, 'Earlier selection, re-timed'),
    'valid': ('o', 3.2, 'white', MUTED, .6, 3, 'Tried, outputs unchanged'),
    'changed': ('x', 3.0, MUTED, MUTED, .6, 3, 'Tried, outputs changed'),
    'memory': ('v', 3.0, 'white', MUTED, .6, 3, 'Tried, over memory limit'),
    'kept': ('o', 3.8, INK, INK, .6, 4, 'Kept'),
}
# What each kept configuration added, and where its label sits: (dx, y factor, ha, va).
KEPT_LABEL = {2: ('Fixed-slot\nGRU', (0, 1.42, 'center', 'bottom')),
              3: ('Decode\nprefetch', (0, 1.28, 'center', 'bottom')),
              4: ('Both\ncombined', (0, 1.13, 'center', 'bottom')),
              5: ('CUDA graphs,\ndetect-ahead', (-.38, 1.0, 'right', 'center')),
              6: ('Four processes', (-.45, 1.0, 'right', 'center')),
              7: ('Eight processes\nunder MPS', (-.38, 1.0, 'right', 'center')),
              11: ('GPU crops,\nblocking waits', (-.3, 1.18, 'right', 'bottom'))}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jitter(name: str, width: float = .62) -> float:
    """Deterministic horizontal spread so runs of one round do not overlap."""
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (h - .5) * width


def build() -> None:
    record = json.loads((DATA / 'prepared.json').read_text())
    for source in record['sources']:
        assert digest(ROOT / source['path']) == source['sha256'], source['path']
    archive, streaming, agents = record['archive'], record['streaming'], record['agents']
    rounds = archive['rounds']
    same_pod = archive['same_pod']
    gpus = streaming['gpus']
    assert len(gpus) == 9 and gpus == sorted(gpus, key=lambda g: -g['images_per_second'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'deepstarnoodles-figure-inference-v2'})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text_in(x, y, s, size=6.5, color=INK, weight='normal', ha='left', **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, ha=ha, va='center', **kw)

    def heading(x, y, letter, title):
        text_in(x, y, letter, 10.5, INK, 'bold')
        text_in(x + .22, y - .005, title, 7.4, INK)   # plain ink; colour only encodes data

    def light_axes(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(GRID)
        ax.tick_params(length=2.2, labelsize=6.0, pad=1.5, colors=MUTED)

    # ------------------------------------------------------------ A: every run of the agent-run rounds
    heading(.08, .15, 'A', 'Agent-run optimization on one RTX 5090: every timed run')
    ax = ax_in(.50, .68, 4.42, 1.98)
    light_axes(ax)
    ax.set_yscale('log')
    ax.set_ylim(2.08, 75)
    ax.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(['3', '5', '10', '20', '50'])
    ax.set_ylabel('Images per second', fontsize=6.3, labelpad=2, color=INK)
    ax.set_xlim(.35, 12.15)
    ax.axhline(2.55, color=GRID, lw=.5, zorder=1)
    ax.text(.42, MEMORY_Y, 'no rate', fontsize=5.2, color=MUTED, ha='left', va='center')

    for rnd in rounds:
        r = rnd['round']
        for run in rnd['runs']:
            marker, size, face, edge, mew, z, _ = STYLE[run['outcome']]
            y = MEMORY_Y if run['outcome'] == 'memory' else run['images_per_second']
            dx = 0 if run['outcome'] == 'kept' else jitter(run['name'])
            ax.plot(r + dx, y, marker, ms=size, mfc=face, mec=edge, mew=mew, zorder=z, ls='none')
    xs = [rnd['round'] for rnd in rounds]
    ys = [rnd['selection_images_per_second'] for rnd in rounds]
    ax.plot(xs, ys, color=INK, lw=.8, zorder=3.5, solid_joinstyle='round')
    for rnd in rounds:
        if rnd['round'] in KEPT_LABEL:
            label, (dx, fy, ha, va) = KEPT_LABEL[rnd['round']]
            ax.text(rnd['round'] + dx, rnd['selection_images_per_second'] * fy, label, fontsize=5.4,
                    color=INK, ha=ha, va=va, linespacing=1.1, zorder=5)
    # Round 10 tested live replay only.
    ax.text(10, 3.1, 'live\nreplay\nonly', fontsize=5.2, color=MUTED, ha='center', va='center', linespacing=1.1)

    ax.set_xticks(range(1, 12))
    ax.set_xticklabels([str(r) for r in range(1, 12)], fontsize=6.0, color=INK)
    ax.tick_params(axis='x', length=0, pad=2.5)
    ax.set_xlabel('Optimization round', fontsize=6.3, labelpad=10, color=INK)
    under = blended_transform_factory(ax.transData, ax.transAxes)
    for rnd in rounds:
        ax.text(rnd['round'], -.115, f"{len(rnd['runs'])}", transform=under, fontsize=5.2, color=MUTED,
                ha='center', va='center')
    ax.text(.35, -.115, 'runs', transform=under, fontsize=5.2, color=MUTED, ha='right', va='center')
    ax.text(10, -.115, '–', transform=under, fontsize=5.2, color=MUTED, ha='center', va='center')

    # Which agent ran which rounds.
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for key in ('codex', 'claude'):
        lo, hi = min(agents[key]['rounds']) - .4, max(agents[key]['rounds']) + .4
        ax.plot([lo, lo, hi, hi], [1.03, 1.06, 1.06, 1.03], color=INK, lw=.6, transform=top, clip_on=False)
        ax.text((lo + hi) / 2, 1.1, agents[key]['label'], transform=top, fontsize=6.0, color=INK,
                ha='center', va='bottom')

    eager, final = same_pod['eager_images_per_second'], same_pod['final_images_per_second']
    bx = 11.75
    ax.plot([bx, bx], [eager, final], color=INK, lw=.6, zorder=2)
    for y in (eager, final):
        ax.plot([bx - .1, bx], [y, y], color=INK, lw=.6, zorder=2)
    ax.text(bx + .1, np.sqrt(eager * final), f'{same_pod["speedup"]:.0f}×\non one\npod', fontsize=5.6,
            color=INK, ha='left', va='center', linespacing=1.1)

    order = ['kept', 'valid', 'changed', 'memory', 'baseline', 'reference']
    handles = [Line2D([], [], marker=STYLE[k][0], ms=STYLE[k][1], mfc=STYLE[k][2], mec=STYLE[k][3], mew=STYLE[k][4],
                      ls='none') for k in order]
    ax.legend(handles, [STYLE[k][6] for k in order], loc='upper left', frameon=False, fontsize=5.6,
              handletextpad=.2, borderaxespad=.3, labelspacing=.35)

    # ------------------------------------------------------------ B: sustained streamed rate by GPU
    rows = np.arange(len(gpus))
    heading(5.30, .15, 'B', 'Deployed pipeline by GPU')
    bx_ = ax_in(6.12, .68, 1.08, 1.98)
    light_axes(bx_)
    rate = np.array([g['images_per_second'] for g in gpus])
    bx_.barh(rows, rate, height=.62, color=BAR, lw=0)
    for r, v in zip(rows, rate):
        bx_.text(v + 1.5, r, f'{v:.1f}', fontsize=5.8, color=INK, ha='left', va='center')
    bx_.set_ylim(len(gpus) - .45, -.55)
    bx_.set_yticks(rows)
    bx_.set_yticklabels([g['label'] for g in gpus], fontsize=6.0, color=INK)
    bx_.tick_params(axis='y', length=0, pad=2.5)
    bx_.spines['left'].set_visible(False)
    bx_.set_xlim(0, 80)
    bx_.set_xticks([0, 20, 40, 60])
    bx_.set_xlabel('Video frames per second', fontsize=6.3, labelpad=10, color=INK)

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles figure: inference throughput', 'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)
    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'inputs': {'prepared.json': digest(DATA / 'prepared.json')},
        'display': {'a_axis': 'log images per second (warmed processing)', 'a_points': 'one marker per timed run',
                    'a_jitter': 'deterministic, from SHA-256 of the run name, width 0.62 round',
                    'a_line': 'selection after each round, as measured on that round\'s pod',
                    'a_memory_strip_y': MEMORY_Y, 'b_session': 'faster of two process counts per GPU',
                    'gpu_order': 'descending images per second'},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
