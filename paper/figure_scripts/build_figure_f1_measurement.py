#!/usr/bin/env python3
"""Render how arm F1 is measured, on one verified test-readout image (tracer, seed 0).

Reads the pinned readout row and test cache for one image, re-runs the
evaluator's matching (which must reproduce the stored counts and Chamfer
values).
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
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np

from prepare_figure_gru_vs_dijkstra_data import MATCH_TOLERANCE_PX, chamfer, match_arms, resample

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
READOUT = ROOT / ('paper/deepstarnoodles_methods/plos/noodle_rerun_20261026/execution/analysis/'
                  'production_readout_v2_20261103/recovery_attempt_001')
VERIFIED = ROOT / ('paper/deepstarnoodles_methods/plos/noodle_rerun_20261026/execution/analysis/'
                   'final_readout_reader_contingency_20261104/verified_full_result.json')
STEM = 'figure_f1_measurement'
IMAGE_ID = 'task_2437521_frame_000006'   # tracer seed 0: matched arms, one near miss (FP + FN) and one missed arm
W, H = 7.48, 4.32
INK, MUTED, GRID = '#20303c', '#5d6d79', '#d8e0e4'
MATCHED, FALSE_POS, STROKE = '#00e4ff', '#FFD700', '#111111'
VIEW_MARGIN_PX = 10
ZOOM_PX = 40
TEST_IMAGES = 553
TYPICAL_HALF_PX = 22


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_example():
    request = READOUT / 'requests/seed0_full_h128'
    pin = json.loads((request / 'complete.json').read_text())['rows']
    assert digest(request / 'rows.jsonl') == pin['sha256'], 'Changed readout rows'
    row, totals, distances = None, np.zeros(3, dtype=int), []
    with open(request / 'rows.jsonl') as stream:          # every test image: counts and matched-arm distances
        for line in stream:
            r = json.loads(line)
            totals += (r['tp'], r['fp'], r['fn']); distances += r['chamfer']
            row = r if r['image_id'] == IMAGE_ID else row
    verified = json.loads(VERIFIED.read_text())['counts']
    assert tuple(totals) == tuple(sum(c[k] for c in verified if c['seed'] == 0 and c['condition'] == 'full_h128')
                                  for k in ('tp', 'fp', 'fn')), 'Totals differ from the verified readout'
    record = next(r for r in json.loads((READOUT / 'fields/seed0_parent/cache/test_manifest.json').read_text())['records']
                  if r['uid'] == IMAGE_ID)
    cache = READOUT / 'fields/seed0_parent/cache' / record['cache_file']
    assert digest(cache) == record['cache_sha256'], 'Changed test cache'
    with np.load(cache, allow_pickle=False) as archive:
        image = archive['image']
    decoded = [np.asarray(a, dtype=np.float32) for a in row['predictions']]
    truth = [np.asarray(a, dtype=np.float32) for a in row['ground_truth']]
    matches = match_arms(decoded, truth)
    assert (len(matches), len(decoded) - len(matches), len(truth) - len(matches)) == (row['tp'], row['fp'], row['fn'])
    assert np.allclose([c for *_, c in matches], row['chamfer'], atol=1e-9), 'Matching differs from the readout'
    return row, image, decoded, truth, matches, totals, np.asarray(distances), pin['sha256'], record['cache_sha256']


def nearest_links(a, b):
    """Each point of a joined to its nearest point of b, and the distances."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    k = d.argmin(1)
    return np.stack([a, b[k]], 1), d[np.arange(len(a)), k]


def build() -> None:
    row, image, decoded, truth, matches, totals, distances, rows_sha, cache_sha = load_example()
    matched_d = {i: (j, c) for i, j, c in matches}
    matched_t = {j for _, j, _ in matches}
    false_pos = [i for i in range(len(decoded)) if i not in matched_d]
    missed = [j for j in range(len(truth)) if j not in matched_t]
    # Near miss: the false positive and the unmatched annotated arm closest to it.
    near_fp, near_gt, near_c = min(((i, j, chamfer(resample(decoded[i]), truth[j])) for i in false_pos for j in missed),
                                   key=lambda t: t[2])
    # A typical match: the matched pair whose distance is nearest this image's median, away from the near miss.
    median_c = float(np.median([c for *_, c in matches]))
    near_centre = np.concatenate([decoded[near_fp], truth[near_gt]]).mean(0)
    apart = {i: v for i, v in matched_d.items()
             if np.abs(np.concatenate([decoded[i], truth[v[0]]]).mean(0) - near_centre).max() > 2 * ZOOM_PX}
    good_i, (good_j, good_c) = min(apart.items(), key=lambda kv: abs(kv[1][1] - median_c))
    tp, fp, fn = row['tp'], row['fp'], row['fn']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'mathtext.fontset': 'dejavusans',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.hashsalt': 'deepstarnoodles-figure-f1-measurement-v2'})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def ax_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)

    def text_in(x, y, s, size=7, color=INK, weight='normal', ha='left', va='center', **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)

    def heading(x, y, number, title):
        text_in(x, y, number, 10.5, INK, 'bold')
        text_in(x + .2, y - .005, title, 7.8, INK)

    def image_ax(x, y, size, view, alpha=1.0):
        ax = ax_in(x, y, size, size)
        ax.imshow(image, interpolation='lanczos', alpha=alpha)
        ax.set(xlim=view[:2], ylim=view[2:][::-1], xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def stroked(ax, xy, color, lw=.9, dashed=False, zorder=5):
        line, = ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle='round', zorder=zorder,
                        ls=(0, (2.2, 1.4)) if dashed else '-')
        line.set_path_effects([patheffects.Stroke(linewidth=lw + .9, foreground=STROKE), patheffects.Normal()])

    def label(ax, x, y, s):
        t = ax.text(x, y, s, fontsize=6.4, color='white', ha='left', va='top', zorder=9)
        t.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground=STROKE)])

    def key(x, y, color, dashed):
        k = ax_in(x, y - .05, .24, .1, facecolor='#4a5862')
        k.set(xlim=(0, 1), ylim=(-1, 1), xticks=[], yticks=[])
        for spine in k.spines.values():
            spine.set_visible(False)
        stroked(k, np.array([[.12, 0.], [.88, 0.]]), color, dashed=dashed)

    def zoom_view(*arms):
        c = np.concatenate(arms).mean(0)
        return (c[0] - ZOOM_PX, c[0] + ZOOM_PX, c[1] - ZOOM_PX, c[1] + ZOOM_PX)

    zooms = [('i', zoom_view(decoded[good_i], truth[good_j]), good_i, good_j, good_c),
             ('ii', zoom_view(decoded[near_fp], truth[near_gt]), near_fp, near_gt, near_c)]

    # 1: distance between a decoded arm and an annotated arm.
    heading(.04, .17, '1', 'Measure how far apart two arms are')
    size = 1.12
    for k, (name, z, i, j, c) in enumerate(zooms):
        x = .12 + k * (size + .12)
        ax = image_ax(x, .42, size, z, alpha=.55)
        a = resample(decoded[i])                              # decoded points every 2 px, as the evaluator does
        for links in (nearest_links(a, truth[j])[0], nearest_links(truth[j], a)[0]):
            for seg in links:
                ax.plot(seg[:, 0], seg[:, 1], color='white', lw=.35, alpha=.8, zorder=4)
        stroked(ax, truth[j], 'white', lw=.8, zorder=5)
        stroked(ax, a, MATCHED if c <= MATCH_TOLERANCE_PX else FALSE_POS, lw=.8, zorder=6)
        label(ax, z[0] + 4, z[2] + 4, name)
        same = c <= MATCH_TOLERANCE_PX
        text_in(x + size / 2, .42 + size + .14, f'{c:.1f} px', 8.0, INK, 'bold', ha='center')
        text_in(x + size / 2, .42 + size + .32, 'same arm' if same else 'too far: no match', 6.8, INK, ha='center')
    text_in(.12, 2.02, f'Distance = average gap between the decoded arm\n(colour) and the annotated arm (white). '
            f'{MATCH_TOLERANCE_PX:.0f} px or\nless counts as the same arm.', 6.6, MUTED, va='top', linespacing=1.4)

    # 2: pair up arms and count.
    heading(2.64, .17, '2', 'Pair arms up, then count')
    everything = np.concatenate(truth + decoded)
    lo, hi = everything.min(0), everything.max(0)
    side = float((hi - lo).max()) + 2 * VIEW_MARGIN_PX
    cx, cy = (lo + hi) / 2
    big = 1.95
    ax = image_ax(2.72, .42, big, (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2))
    for j in missed:
        stroked(ax, truth[j], 'white', dashed=True, zorder=4)
    for i, arm in enumerate(decoded):
        stroked(ax, arm, MATCHED if i in matched_d else FALSE_POS)
    for name, z, *_ in zooms:
        ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='white', lw=.6, zorder=7,
                               path_effects=[patheffects.Stroke(linewidth=1.4, foreground=STROKE), patheffects.Normal()]))
        label(ax, z[0] + 3, z[2] + 3, name)
    for k, (count, color, dashed, name) in enumerate(((tp, MATCHED, False, 'matched (TP)'),
                                                      (fp, FALSE_POS, False, 'extra (FP)'),
                                                      (fn, 'white', True, 'missed (FN)'))):
        y = .62 + k * .26
        key(4.80, y, color, dashed)
        text_in(5.30, y, f'{count}', 8.0, INK, 'bold', ha='right')
        text_in(5.37, y, name, 7.0, INK)
    text_in(4.80, 1.50, 'Closest pairs first;\neach arm is used once.', 6.6, MUTED, va='top', linespacing=1.4)

    # 3: the score.
    heading(6.08, .17, '3', 'Score')
    text_in(6.16, .80, r'$F_1 = \frac{2\,TP}{2\,TP + FP + FN}$', 10.5, INK)
    text_in(6.16, 1.32, f'$= \\frac{{{2 * tp}}}{{{2 * tp} + {fp} + {fn}}} = {2 * tp / (2 * tp + fp + fn):.2f}$', 10.5, INK)
    text_in(6.16, 1.72, '1 = every arm found\nand nothing extra', 6.6, MUTED, va='top', linespacing=1.4)

    # The model on the whole test set: a typical matched arm, and all matched-arm distances.
    tp_all, fp_all, fn_all = (int(v) for v in totals)
    f1_all = 2 * tp_all / (2 * tp_all + fp_all + fn_all)
    median_all = float(np.median(distances))
    text_in(.04, 2.66, f'Our model on the test set ({TEST_IMAGES} images)', 7.8, INK, 'bold')
    typical_i, (typical_j, typical_c) = min(matched_d.items(), key=lambda kv: abs(kv[1][1] - median_all))
    arm_d, arm_t = resample(decoded[typical_i]), truth[typical_j]
    centre = arm_t[len(arm_t) // 2] - (8, 0)                # arm right of centre, leaving room for the scale bars
    half = TYPICAL_HALF_PX
    tz = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
    size_t = 1.22
    ax = image_ax(.12, 2.92, size_t, tz, alpha=.55)
    for links in (nearest_links(arm_d, arm_t)[0], nearest_links(arm_t, arm_d)[0]):
        for seg in links:
            ax.plot(seg[:, 0], seg[:, 1], color='white', lw=.35, alpha=.8, zorder=4)
    stroked(ax, arm_t, 'white', lw=.8, zorder=5)
    stroked(ax, arm_d, MATCHED, lw=.8, zorder=6)
    bx = tz[0] + 3
    for k, (length, name) in enumerate(((median_all, f'{median_all:.1f} px median'),
                                        (MATCH_TOLERANCE_PX, f'{MATCH_TOLERANCE_PX:.0f} px limit'))):
        y = tz[3] - 3 - (1 - k) * 10
        ax.plot([bx, bx + length], [y, y], color='white', lw=1.6, solid_capstyle='butt', zorder=8,
                path_effects=[patheffects.Stroke(linewidth=2.6, foreground=STROKE), patheffects.Normal()])
        t = ax.text(bx, y - 1.6, name, fontsize=5.8, color='white', va='bottom', zorder=9)
        t.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground=STROKE)])
    text_in(.12 + size_t + .1, 3.18, 'A typical fit', 6.8, INK, 'bold')
    text_in(.12 + size_t + .1, 3.38, f'This arm is {typical_c:.1f} px\nfrom its annotation.\n\nMedian over all\n'
            f'matched arms: {median_all:.1f} px.', 6.3, INK, va='top', linespacing=1.35)

    ax = ax_in(3.05, 2.92, 2.05, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    ax.tick_params(length=2.2, labelsize=6.2, pad=1.5, colors=MUTED)
    ax.hist(distances, bins=np.arange(0, MATCH_TOLERANCE_PX + .01, .25), color='#9aa6af', lw=0)
    ax.axvline(median_all, color=INK, lw=1.0)
    ax.text(median_all + .15, ax.get_ylim()[1] * .97, f'median {median_all:.1f} px', fontsize=6.4, color=INK, va='top')
    ax.set_xlim(0, MATCH_TOLERANCE_PX)
    ax.set_xlabel(f'Distance of each of the {tp_all:,} matched arms (px)', fontsize=6.5, labelpad=2, color=INK)
    ax.set_ylabel('Arms', fontsize=6.5, labelpad=2, color=INK)

    for k, (count, color, dashed, name) in enumerate(((tp_all, MATCHED, False, 'matched (TP)'),
                                                      (fp_all, FALSE_POS, False, 'extra (FP)'),
                                                      (fn_all, 'white', True, 'missed (FN)'))):
        y = 3.0 + k * .24
        key(5.55, y, color, dashed)
        text_in(6.22, y, f'{count:,}', 8.0, INK, 'bold', ha='right')
        text_in(6.29, y, name, 7.0, INK)
    n = lambda v: f'{v:,}'.replace(',', '{,}')               # mathtext comma without the list space
    text_in(5.55, 3.90, f'$F_1 = \\frac{{{n(2 * tp_all)}}}{{{n(2 * tp_all)} + {n(fp_all)} + {n(fn_all)}}} = {f1_all:.2f}$',
            10.0, INK)

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles: how arm F1 is measured', 'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)
    provenance = {
        'figure': STEM, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(Path(__file__)),
        'inputs': {'readout_rows_seed0_full_h128_sha256': rows_sha, 'cache_sha256': cache_sha,
                   'verified_result_sha256': digest(VERIFIED)},
        'test_set': {'images': TEST_IMAGES, 'tp': tp_all, 'fp': fp_all, 'fn': fn_all, 'pooled_f1': f1_all,
                     'median_matched_distance_px': median_all,
                     'quartiles_px': np.percentile(distances, [25, 75]).tolist()},
        'example': {'image_id': IMAGE_ID, 'seed': 0, 'condition': 'full_h128', 'tp': tp, 'fp': fp, 'fn': fn,
                    'zoom_i': {'decoded': good_i, 'ground_truth': good_j, 'chamfer_px': good_c},
                    'zoom_ii': {'decoded': near_fp, 'ground_truth': near_gt, 'chamfer_px': near_c}},
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    print('wrote', STEM, provenance['example'])


if __name__ == '__main__':
    build()
