#!/usr/bin/env python3
"""Render Supplementary Figure S1 (v3): the StarSeg label-granularity experiment, 60-fit retrain.

Layout is the original S1 (build_figure_s1.py), adapted panel by panel to the data of the
2026-11-06 retrain. Reads only figure_s1_v2_data/prepared.json (prepare_figure_s1_v2.py, which
checks every tp/fp/fn bit-for-bit against the campaign's per_run_metrics.csv). All scheme
comparisons are at model size L; dots are the three seeds, bars their mean.

  A  training and test instance masks by domain, stacked by coarse class
  B  sunflower-star precision, recall and F1 on test images outside the centred field photos
  C  precision and recall by domain
  D  masks per annotated class and the labels each scheme trains on
  E  confusion matrix of scheme F on test, pooled over seeds
  F  % of each class's test objects predicted as sunflower star, per scheme
NumPy and matplotlib only.
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
DATA = HERE / 'figure_s1_v2_data'
STEM = 'figure_s1_v3_starseg'
W, H = 7.48, 9.9
INK, MUTED, GRID, LINE = '#20303c', '#5d6d79', '#d8e0e4', '#c9d1d7'
# Coarse classes: re-stepped Figure 3 hues, validated (all pairs, light surface).
COARSE = {'sunflower_star': ('Sunflower star', '#D9453F'),
          'other_sea_star': ('Other sea stars', '#2F6DB5'),
          'live_prey': ('Live prey', '#EC9A2B'),
          'dead_prey': ('Dead prey', '#1C9C8C')}
# Schemes are ordered by granularity, so they get one neutral light-to-dark ramp.
SCHEMES = [('F', 37, '#1f2d38'), ('C', 4, '#4f6170'), ('B', 2, '#8597a4'), ('P', 1, '#b9c4cc')]
SCHEME_NAMES = {'F': 'Fine', 'C': 'Coarse', 'B': 'Binary', 'P': 'Sunflower only'}
SIZE = 'l'          # every scheme comparison is at YOLO26-L
B_STRATUM = 'all_excl_centred'
SUN = 'Pycnopodia_helianthoides'
SEQ = LinearSegmentedColormap.from_list('slate', ['#ffffff', '#c9d6df', '#6f8797', '#1f2d38'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pretty(name):
    special = {'Unknown_species': 'Unknown sea star', 'Unknown_urchin': 'Unknown urchin',
               'Bivalve': 'Bivalve', 'Dead_bivalve': 'Dead bivalve', 'Dead_urchin': 'Dead urchin',
               'Dead_sandDollar': 'Dead sand dollar', 'Sand_dollar': 'Sand dollar'}
    if name in special:
        return special[name], False
    genus, species = name.split('_', 1)
    return f'{genus} {species}', True


def build():
    rec = json.loads((DATA / 'prepared.json').read_text())
    assert rec['verification']['all_passed'] and rec['verification']['mismatches'] == 0
    op = rec['operating_point']
    assert op['confidence'] == 0.25 and op['mask_iou'] == 0.5
    fine = rec['fine_classes']
    coarse_of = rec['fine_to_coarse']
    counts = {r['fine_label']: r for r in rec['class_counts']}
    order = sorted(fine, key=lambda n: (list(COARSE).index(coarse_of[n]), -counts[n]['train_masks']))
    assert len(order) == 37 and order[0] == SUN
    n = len(order)
    per_run = rec['per_run']
    cfg = {(r['size'], r['scheme'], r['stratum']): r for r in rec['configuration']}
    DOMAINS = [(d['key'], d['label']) for d in rec['domains']]
    assert [k for k, _ in DOMAINS] == ['lab', 'field_survey', 'captive', 'field_centred', 'web']
    dom_rows = {(r['domain'], r['split']): r for r in rec['domain_counts']}

    def seed_rows(scheme, stratum):
        rows = sorted((r for r in per_run if r['size'] == SIZE and r['scheme'] == scheme
                       and r['stratum'] == stratum), key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == [0, 1, 2], (scheme, stratum)
        return rows

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.2, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'deepstarnoodles-figure-s1-v3',
                         'axes.edgecolor': LINE, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.linewidth': .5, 'xtick.major.width': .5,
                         'ytick.major.width': .5, 'xtick.major.size': 2, 'ytick.major.size': 2})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text_in(x, y, s, size=6.2, color=INK, weight='normal', ha='left', va='center', **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)

    def heading(x, y, letter, title):
        text_in(x, y, letter, 10, INK, 'bold')
        text_in(x + .2, y, title, 7.2, INK)

    def tidy(ax):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=5.6, pad=1.5)

    # ---- A: data by domain ------------------------------------------------
    heading(.05, .2, 'A', 'Training and test data by domain')
    bx = ax_in(.75, .48, 2.35, 1.22)
    ys = np.arange(len(DOMAINS))[::-1]
    for y, (dom, label) in zip(ys, DOMAINS):
        tr, te = dom_rows[(dom, 'train')], dom_rows[(dom, 'test')]
        # Two thin bars per domain: training (solid) and test (tint), stacked by coarse class.
        for offset, src, alpha, tag in ((.17, tr, 1.0, 'train'), (-.17, te, .45, 'test')):
            left = 0
            for cls, (cname, color) in COARSE.items():
                v = src.get(cls, 0)
                if v:
                    bx.barh(y + offset, v, left=left, height=.3, color=color, alpha=alpha,
                            edgecolor='white', linewidth=.6)
                left += v
            assert left == src['instances']
            bx.text(left + 150, y + offset, f"{tag} · {src['images']:,} img",
                    fontsize=4.9, color=MUTED, va='center')
    # the original's label margin fits one line of about 0.73 in; the two longer names wrap
    wrap = {'Captive non-lab tanks': 'Captive\nnon-lab tanks', 'Centred field photos': 'Centred\nfield photos'}
    bx.set_yticks(ys, [wrap.get(label, label) for _, label in DOMAINS], fontsize=5.8, color=INK)
    bx.set_ylim(-.5, len(DOMAINS) - .5)
    bx.set_xlim(0, 21000)
    bx.set_xticks([0, 5000, 10000, 15000, 20000])
    bx.set_xlabel('Instance masks', fontsize=5.8, labelpad=1)
    bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v/1000)}k' if v else '0'))
    tidy(bx)
    bx.tick_params(axis='y', length=0)
    for i, (cls, (cname, color)) in enumerate(COARSE.items()):
        x0 = (.75, 1.55, 2.4, 3.02)[i]
        fig.add_artist(Rectangle((x0 / W, 1 - 2.1 / H), .07 / W, .07 / H, color=color, transform=fig.transFigure))
        text_in(x0 + .1, 2.065, cname, 5.4, INK)

    # ---- B: sunflower precision, recall, F1 (centred photos excluded) ------
    heading(3.55, .2, 'B', 'Sunflower-star detection on test images (centred photos excluded)')
    for k, (s, ncls, color) in enumerate(SCHEMES):
        lx = (3.85, 4.72, 5.6, 6.42)[k]
        fig.add_artist(Line2D([(lx + .03) / W], [1 - 2.065 / H], marker='o', ms=3.6, mfc=color,
                              mec=INK if s == 'P' else 'white', mew=.4, ls='none', transform=fig.transFigure))
        text_in(lx + .1, 2.065, f'{s}  {SCHEME_NAMES[s]} ({ncls})', 5.4, INK)
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]
    xs = np.arange(len(SCHEMES))
    rng = np.random.default_rng(3)
    jit = {s: rng.uniform(-.13, .13, 3) for s, _, _ in SCHEMES}

    def seed_panel(ax, stratum, key, ylim, offsets=None):
        for i, (s, ncls, color) in enumerate(SCHEMES):
            rows = seed_rows(s, stratum)
            vals = np.array([r[key] for r in rows])
            dx = jit[s] if offsets is None else offsets
            ax.plot([i - .26, i + .26], [vals.mean()] * 2, color=color, lw=1.6, solid_capstyle='round', zorder=2)
            ax.scatter(i + dx, vals, s=9, color=color, edgecolor=INK if s == 'P' else 'white',
                       linewidth=.4, zorder=3)
        ax.set_xticks(xs, [f'{s}\n{ncls}' for s, ncls, _ in SCHEMES], fontsize=5.6)
        ax.set_xlim(-.6, len(SCHEMES) - .4)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color=GRID, lw=.4)
        ax.set_axisbelow(True)
        tidy(ax)

    b_ref = cfg[(SIZE, 'F', B_STRATUM)]
    for j, (key, label) in enumerate(metrics):
        ax = ax_in(3.85 + j * 1.2, .48, .95, 1.22)
        vals = [r[key] for s, _, _ in SCHEMES for r in seed_rows(s, B_STRATUM)]
        for s, _, _ in SCHEMES:
            assert all(r['images'] == b_ref['images'] and r['sunflower_stars'] == b_ref['sunflower_stars']
                       for r in seed_rows(s, B_STRATUM))
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * .12
        seed_panel(ax, B_STRATUM, key, (lo - pad, hi + pad))
        ax.set_title(label, fontsize=6, color=INK, pad=2)

    # ---- C: precision and recall by domain --------------------------------
    heading(.05, 2.38, 'C', 'Precision and recall at the runtime threshold (confidence 0.25, mask IoU ≥ 0.5)')
    # Five blocks in the original's single row: narrower subpanels, same 2.82 in top and 1.0 in height.
    # The unit ('sunflower stars') moves to the caption so each block title stays on one line.
    # Seeds sit at fixed offsets so tied values (e.g. three seeds at precision 1) stay distinguishable.
    bstep, aw, astep = 1.5, .48, .67
    for d, (dom, label) in enumerate(DOMAINS):
        x0 = .27 + d * bstep
        gt = cfg[(SIZE, 'F', dom)]['sunflower_stars']
        text_in(x0 + (astep + aw) / 2, 2.62, f'{label}  (n = {gt:,})', 6, INK, ha='center')
        for k, (metric, mlabel) in enumerate((('precision', 'Precision'), ('recall', 'Recall'))):
            ax = ax_in(x0 + k * astep, 2.82, aw, 1.0)
            seed_panel(ax, dom, metric, (0, 1.04), offsets=np.array([-.22, 0, .22]))
            ax.set_yticks([0, .5, 1])
            ax.set_title(mlabel, fontsize=5.8, color=MUTED, pad=1.5)
    # the five domains partition the test set
    for s, _, _ in SCHEMES:
        parts = [cfg[(SIZE, s, k)] for k, _ in DOMAINS]
        whole, excl, cent = cfg[(SIZE, s, 'all')], cfg[(SIZE, s, B_STRATUM)], cfg[(SIZE, s, 'field_centred')]
        for key in ('images', 'sunflower_stars'):
            assert whole[key] == sum(p[key] for p in parts) == excl[key] + cent[key]

    # ---- per-class block: D | label scheme | E | F --------------------------
    top, row_h = 5.3, .108
    y_of = {name: top + (i + .5) * row_h for i, name in enumerate(order)}
    bottom = top + n * row_h
    heading(.05, 4.2, 'D', 'Masks per annotated class')
    heading(3.48, 4.2, 'E', 'Confusion matrix, Fine model (F, 37 classes)')
    text_in(6.36, 4.2, 'F', 10, INK, 'bold')
    text_in(6.55, 4.2, 'Predicted as', 7.2, INK)
    text_in(6.55, 4.37, 'sunflower star (%)', 7.2, INK)

    # Row labels, grouped by coarse class.
    for name in order:
        label, italic = pretty(name)
        text_in(1.36, y_of[name], label, 4.6, INK, ha='right', style='italic' if italic else 'normal')
    groups = {}
    for name in order:
        groups.setdefault(coarse_of[name], []).append(name)
    for cls, members in groups.items():
        y0, y1 = y_of[members[0]] - row_h / 2, y_of[members[-1]] + row_h / 2
        fig.add_artist(Rectangle((.06 / W, 1 - y1 / H), .045 / W, (y1 - y0) / H, color=COARSE[cls][1],
                                 transform=fig.transFigure))
        if len(members) >= 3:
            text_in(.15, (y0 + y1) / 2, COARSE[cls][0], 5 if len(members) > 3 else 4.5, COARSE[cls][1], 'bold', rotation=90, ha='center')
        else:
            short_name = 'Sunflower' if cls == 'sunflower_star' else COARSE[cls][0]
            text_in(.14, (y0 + y1) / 2, short_name, 4.8, COARSE[cls][1], 'bold')
        if cls != 'sunflower_star':
            fig.add_artist(Line2D([.12 / W, 7.42 / W], [1 - y0 / H] * 2, color=LINE, lw=.4,
                                  transform=fig.transFigure))

    # D: train (filled) and test (open) mask counts, log scale.
    ax = ax_in(1.42, top, 1.2, n * row_h)
    for y, name in enumerate(order):
        c = counts[name]
        color = COARSE[coarse_of[name]][1]
        tr, te = c['train_masks'], c['test_masks']
        ax.plot([min(tr, te or tr), tr], [y, y], color=GRID, lw=.6, zorder=1)
        ax.scatter(tr, y, s=7, color=color, edgecolor='white', linewidth=.3, zorder=3)
        if te:
            ax.scatter(te, y, s=7, facecolor='white', edgecolor=color, linewidth=.6, zorder=3, clip_on=False)
        else:
            ax.text(1.2, y, '–', fontsize=4.5, color=MUTED, va='center')
    ax.set_xscale('log')
    ax.set_xlim(1, 15000)
    ax.set_ylim(n - .5, -.5)
    ax.set_yticks([])
    ax.set_xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1k', '10k'])
    ax.minorticks_off()
    ax.xaxis.grid(True, color=GRID, lw=.4)
    ax.set_axisbelow(True)
    tidy(ax)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Masks (log)', fontsize=5.6, labelpad=1)
    ax.scatter([], [], s=7, color=MUTED, label='training')
    ax.scatter([], [], s=7, facecolor='white', edgecolor=MUTED, linewidth=.6, label='test')
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, -.005), fontsize=4.8, frameon=False,
              handletextpad=.1, borderaxespad=0, labelspacing=.25)

    # Scheme glyph: which labels each scheme trains on.
    gx0, gw = 2.78, .14
    for j, (s, ncls, _) in enumerate(SCHEMES):
        assert rec['scheme_classes'][s] == ncls
        x = gx0 + j * (gw + .04)
        text_in(x + gw / 2, top - .06, f'{SCHEME_NAMES[s]} ({s})', 5.2, INK, ha='center', va='bottom', rotation=90)
        text_in(x + gw / 2, bottom + .1, str(ncls), 5.2, MUTED, ha='center')
        for i, name in enumerate(order):
            cls = coarse_of[name]
            y0 = top + i * row_h
            if s == 'P' and cls != 'sunflower_star':
                color, alpha = None, 0
            elif s == 'B' and cls != 'sunflower_star':
                color, alpha = '#8597a4', 1
            else:
                color, alpha = COARSE[cls][1], 1
            if color:
                # Fine: every row its own block; coarse/binary: contiguous blocks per class.
                same_next = (i + 1 < n and s != 'F' and
                             (coarse_of[order[i + 1]] == cls if s == 'C' else
                              (coarse_of[order[i + 1]] != 'sunflower_star') == (cls != 'sunflower_star')))
                gap = 0 if same_next else .018
                fig.add_artist(Rectangle((x / W, 1 - (y0 + row_h - gap) / H), gw / W, (row_h - gap) / H,
                                         color=color, alpha=alpha, lw=0, transform=fig.transFigure))

    # E: confusion of scheme F (size L), pooled over seeds, row-normalized.
    cols = order + ['__missed__']
    idx = {c: i for i, c in enumerate(cols)}
    mat = np.zeros((n, len(cols)))
    background_fp = {}
    for r in rec['confusion_l_F_pooled']:
        if r['true'] == '__background__':  # predictions that matched no annotated object
            background_fp[r['predicted']] = r['count_pooled_3_seeds']
            continue
        mat[order.index(r['true']), idx[r['predicted']]] += r['count_pooled_3_seeds']
    totals = mat.sum(1, keepdims=True)
    for i, name in enumerate(order):  # every test object is either matched or missed
        assert totals[i, 0] == 3 * counts[name]['test_masks'], name
    frac = np.divide(mat, totals, out=np.zeros_like(mat), where=totals > 0)
    pas = rec['predicted_as_sunflower_l_pooled']
    assert mat[0, 0] == pas['F'][SUN]['predicted_sunflower']
    assert background_fp.get(SUN, 0) == pas['F']['__background__']['predicted_sunflower']
    ex0, ew = 3.48, 2.8
    ax = ax_in(ex0, top, ew, n * row_h)
    shown = np.ma.masked_where(frac == 0, frac)
    ax.imshow(shown, cmap=SEQ, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_xlim(-.5, len(cols) - .5)
    ax.set_ylim(n - .5, -.5)
    for i in range(n):
        if totals[i, 0] == 0:  # no test examples of this class
            ax.text(i, i, '–', fontsize=4.5, color=MUTED, ha='center', va='center')
        else:
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=.35))
    ax.axvline(n - .5, color=INK, lw=.5)
    ax.set_xticks(range(len(cols)))

    def short(c):
        label, italic = pretty(c)
        return f"{label.split()[0][0]}. {label.split()[1]}" if italic else label
    ax.set_xticklabels([('Missed' if c == '__missed__' else short(c)) for c in cols], rotation=90,
                       fontsize=4.3)
    for lab, c in zip(ax.get_xticklabels(), cols):
        if c != '__missed__' and pretty(c)[1]:
            lab.set_style('italic')
        if c != '__missed__':
            lab.set_color(COARSE[coarse_of[c]][1])
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set(color=LINE, lw=.4)
    text_in(ex0 + ew / 2, bottom + .1, 'Predicted class', 5.6, INK, ha='center')
    # colorbar
    cax = ax_in(ex0 + .5, bottom + .22, 1.8, .06)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.Normalize(0, 100)),
                      cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(.3)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_ticks([0, 25, 50, 75, 100])
    text_in(ex0 + .45, bottom + .25, '%', 5, MUTED, ha='right')

    # F: % of each class's test objects predicted as sunflower star, per scheme (size L), pooled over seeds.
    fx0, fw = 6.4, .25
    for j, (s, ncls, _) in enumerate(SCHEMES):
        x = fx0 + j * fw
        text_in(x + (fw - .02) / 2, top - .06, f'{SCHEME_NAMES[s]} ({s})', 5.2, INK, ha='center', va='bottom', rotation=90)
        for i, name in enumerate(order):
            p = pas[s][name]
            assert p['objects_x3'] == 3 * counts[name]['test_masks']
            if p['objects_x3'] == 0:
                text_in(x + (fw - .02) / 2, top + (i + .5) * row_h, '–', 4.5, MUTED, ha='center')
                continue
            v = p['predicted_sunflower'] / p['objects_x3']
            y0 = top + i * row_h
            fig.add_artist(Rectangle((x / W, 1 - (y0 + row_h) / H), (fw - .02) / W, (row_h - .012) / H,
                                     color=SEQ(v), lw=0, transform=fig.transFigure))
            if v >= .005:
                text_in(x + (fw - .02) / 2, y0 + row_h / 2, f'{100 * v:.0f}', 4.3,
                        'white' if v > .55 else INK, ha='center')
        # sunflower predictions that matched no annotated object (false alarms on background)
        text_in(x + (fw - .02) / 2, bottom + .3, f"{pas[s]['__background__']['predicted_sunflower']}",
                4.6, INK, ha='center')
    text_in(fx0 + 2 * fw - .01, bottom + .06, 'false alarms on', 4.8, MUTED, ha='center', va='top')
    text_in(fx0 + 2 * fw - .01, bottom + .14, 'background (n)', 4.8, MUTED, ha='center', va='top')

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles Supplementary Figure S1: StarSeg label granularity',
                                  'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)

    def summary(stratum):
        return {s: {k: cfg[(SIZE, s, stratum)][k] for k in
                    ('images', 'sunflower_stars', 'precision_mean', 'recall_mean', 'f1_mean')}
                for s, _, _ in SCHEMES}

    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'layout_from': 'build_figure_s1.py (original S1)',
        'inputs': {'figure_s1_v2_data/prepared.json': digest(DATA / 'prepared.json')},
        'prepare_script_sha256_recorded_in_prepared_json': rec['preparation_script_sha256'],
        'verification (from prepared.json)': rec['verification'],
        'operating_point': op,
        'size': SIZE,
        'values_shown': {
            'B_' + B_STRATUM: summary(B_STRATUM),
            'C': {k: summary(k) for k, _ in DOMAINS},
            'C_sunflower_predictions_per_seed_field_survey': {s: [r['pred'] for r in seed_rows(s, 'field_survey')]
                                                              for s, _, _ in SCHEMES},
            'F_percent_sunflower_row': {s: pas[s][SUN]['percent'] for s, _, _ in SCHEMES},
            'F_background_false_alarms_pooled': {s: pas[s]['__background__']['predicted_sunflower']
                                                 for s, _, _ in SCHEMES},
        },
        'display': {'confusion_scheme': 'F', 'confusion_pooled_over_seeds': [0, 1, 2],
                    'coarse_palette': {k: v[1] for k, v in COARSE.items()},
                    'scheme_ramp': {s: c for s, _, c in SCHEMES}},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
