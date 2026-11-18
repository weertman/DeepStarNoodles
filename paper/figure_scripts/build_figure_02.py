#!/usr/bin/env python3
"""Render Figure 2 from fresh, recorded inference; never synthesizes model data."""
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
from matplotlib.colors import Normalize, PowerNorm, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon
from matplotlib import patheffects
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE / 'figure_02_data'
W, H = 1600, 1440
INK, MUTED, LINE = '#20303c', '#586976', '#d7dfe3'
TEAL, BLUE, AMBER = '#237b76', '#35668f', '#93651e'
TRACE, SEED = '#00e4ff', '#ffdc59'
GAMMA = .55
SUPPORT_THRESHOLD = .5
ZOOM_MARGIN_PX = 6
STEP_INSET_HALF_PX = 18   # C inset: 36 x 36 crop px around the tip seed (the 33-px window fits)
ANNOTATION_ARROW_FRACTIONS = (.5, 1.0)   # mid-arm, and ending exactly at the tip
MASK_BACKGROUND = '#f2f4f6'
CHANNEL_ICONS = ['support', 'centerline', 'dx', 'dy', 'progress', 'mouth',
                 'madreporite', 'anus', 'base', 'tip'] + ['embedding'] * 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pca_rgb(channels, mask=None, blank_outside=True):
    """Deterministic display-only PCA: signed axes, per-axis 2/98% scaling."""
    x = channels.reshape(channels.shape[0], -1).T.astype(np.float64)
    included = np.ones(len(x), dtype=bool) if mask is None else mask.ravel()
    mean = x[included].mean(0)
    _, _, vt = np.linalg.svd(x[included] - mean, full_matrices=False)
    basis = vt[:3].copy()
    for axis in basis:
        if axis[np.abs(axis).argmax()] < 0:
            axis *= -1
    scores = (x - mean) @ basis.T
    lo, hi = np.percentile(scores[included], [2, 98], axis=0)
    rgb = np.clip((scores - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    if blank_outside:
        rgb[~included] = to_rgb(MASK_BACKGROUND)
    return rgb.reshape(*channels.shape[1:], 3), {'mean': mean.tolist(), 'basis': basis.tolist(),
        'lower_2_percentile': lo.tolist(), 'upper_98_percentile': hi.tolist()}


def build():
    inference = json.loads((DATA / 'inference.json').read_text())
    for name, expected in inference['files'].items():
        assert digest(DATA / name) == expected, f'Changed source: {name}'
    d = np.load(DATA / 'predictions.npz')
    target_record = json.loads((DATA / 'targets.json').read_text())
    for name, expected in target_record['files'].items():
        assert digest(DATA / name) == expected, f'Changed target: {name}'
    assert digest(ROOT / target_record['dataset_source_path']) == target_record['dataset_source_sha256']
    target_data = np.load(DATA / 'targets.npz')
    target, instances = target_data['targets'], target_data['arm_instance']
    traces = json.loads((DATA / 'traces.json').read_text())
    batch = json.loads((DATA / 'batched_rollout.json').read_text())
    for name, expected in batch['inputs'].items():
        assert digest(DATA / name) == expected, f'Changed batch input: {name}'
    assert digest(ROOT / 'CORE/deepstarnoodles/noodle_tracer.py') == batch['decoder_source_sha256']
    row = inference['selection']['selected_row']
    annotation_source = next(s for s in inference['sources'] if s['path'] == row['shapes_path'])
    assert digest(ROOT / row['shapes_path']) == annotation_source['sha256'], 'Changed reference annotation'
    model_source = next(s for s in inference['sources'] if s['path'] == 'CORE/deepstarnoodles/model.py')
    assert digest(ROOT / model_source['path']) == model_source['sha256'], 'Changed dense model contract'
    a, raw, image = d['activated'], d['raw'], d['image_rgb']
    rgb = np.clip(image, 0, 1) ** GAMMA
    mask = a[0] > SUPPORT_THRESHOLD
    embedding, embedding_pca = pca_rgb(raw[10:18], mask, blank_outside=False)
    points = d['positions']
    seed = points[0]
    # One square view for every display of the example: the union of the annotated
    # shapes, predicted support > 0.5, exported traces and the seed's 33-px window,
    # plus a small margin, so the arms stop just short of the edges.
    shapes_for_extent = json.loads((ROOT / row['shapes_path']).read_text())
    scale_xy = np.array([512/int(row['width']), 512/int(row['height'])])
    extent_points = [np.asarray(sh['points']).reshape(-1, 2) * scale_xy
                     for sh in shapes_for_extent if not sh.get('outside')]
    sy_, sx_ = np.nonzero(mask)
    extent_points.append(np.column_stack([sx_, sy_]))
    extent_points += [np.asarray(arm['polyline']) for arm in traces['arms']]
    extent_points.append(np.array([seed - 16.5, seed + 16.5]))
    for walker in batch['walkers']:   # every tip's starting window is drawn in D
        tip_xy = np.asarray(walker['rollout'][0]['positions'])
        extent_points.append(np.array([tip_xy - 16.5, tip_xy + 16.5]))
    lo, hi = np.min([e.min(0) for e in extent_points], 0), np.max([e.max(0) for e in extent_points], 0)
    side = float((hi - lo).max() + 2 * ZOOM_MARGIN_PX)
    centre = (lo + hi) / 2
    zx0, zy0 = np.clip(centre - side / 2, -0.5, 511.5 - side)
    zoom = (float(zx0), float(zx0 + side), float(zy0), float(zy0 + side))
    zoom_axes = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'svg.hashsalt': 'deepstarnoodles-figure-02-v1'})
    fig = plt.figure(figsize=(7.48, 7.48 * H / W), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, W), ylim=(H, 0))
    canvas.axis('off')

    def text(x, y, s, size=7, color=INK, weight='normal', ha='center', **kw):
        return canvas.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                           ha=ha, va='center', linespacing=1.35, **kw)

    def arrow(start, end, color=MUTED, gap_start=True):
        # 1.6-pt clearances keep heads and tails off the boxes and images they join.
        canvas.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8,
                                        color=color, lw=.85, shrinkA=1.6 if gap_start else 0,
                                        shrinkB=1.6))

    def path_arrow(points, color=MUTED):
        if len(points) > 2:
            canvas.plot(*zip(*points[:-1]), color=color, lw=.85,
                        solid_joinstyle='round', zorder=2)
        arrow(points[-2], points[-1], color, gap_start=len(points) == 2)

    def block(x, y, width, height, title, sub='', color=TEAL, fill='#edf6f3', title_size=6.8):
        patch = FancyBboxPatch((x, y), width, height,
                              boxstyle='round,pad=0,rounding_size=5',
                              linewidth=.75, edgecolor=color, facecolor=fill, zorder=3)
        canvas.add_patch(patch)
        text(x + width/2, y + height*(.34 if sub else .5), title,
             title_size, color, weight='bold', zorder=4)
        if sub:
            text(x + width/2, y + height*.73, sub, 5.7, MUTED, zorder=4)
        return patch

    def panel(letter, title, sub, x, color):
        text(x, 35, letter, 10.4, weight='bold', ha='left')
        text(x + 32, 36, title, 7.4, INK, ha='left')   # plain ink; colour only encodes data
        text(x, 77, sub, 6.5, MUTED, ha='left')

    def axis(x, y, width, height=None):
        height = width if height is None else height
        ax = fig.add_axes([x / W, 1 - (y + height) / H, width / W, height / H])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE); spine.set_linewidth(.5)
        return ax

    def icon(ax, x, y, size, kind, badge=False):
        """Shared vector symbols on B labels and C channel patches; y increases down."""
        if badge:
            ax.add_patch(Rectangle((x, y), size, size, fc='white', ec=LINE,
                                   alpha=.96, lw=.25, zorder=9))
        def xy(u, v):
            return x + u*size, y + v*size
        def line(vertices, lw=.8, color=INK):
            ax.plot(*zip(*(xy(*p) for p in vertices)), color=color, lw=lw,
                    solid_capstyle='round', solid_joinstyle='round', zorder=10)
        if kind in ('support', 'centerline'):
            line([(.24,.82),(.36,.57),(.65,.43),(.78,.18)],
                 lw=2.4 if kind == 'support' else .8)
        elif kind in ('dx', 'dy', 'direction'):
            if kind in ('dx', 'direction'):
                line([(.16,.5),(.84,.5)])
                line([(.64,.3),(.84,.5),(.64,.7)])
            if kind in ('dy', 'direction'):
                line([(.5,.16),(.5,.84)])
                line([(.3,.64),(.5,.84),(.7,.64)])
        elif kind == 'progress':
            for i in range(4):
                ax.add_patch(Rectangle(xy(.15+i*.18, .78-i*.17), size*.14,
                                      size*(.12+i*.17), fc=INK, ec='none', zorder=10))
        elif kind in ('mouth', 'madreporite', 'anus'):
            colors = {'mouth':'#aa359e', 'madreporite':'#078ba0', 'anus':'#9e7a00'}
            color = colors[kind]
            if kind == 'madreporite':
                ax.add_patch(Polygon([xy(.5,.12),xy(.88,.5),xy(.5,.88),xy(.12,.5)],
                                     fill=False, ec=color, lw=.9, zorder=10))
            else:
                ax.add_patch(Circle(xy(.5,.5), size*.3,
                                    fc=color if kind == 'anus' else 'none',
                                    ec=color, lw=.9, zorder=10))
        elif kind in ('base', 'tip'):
            if kind == 'base':
                ax.add_patch(Rectangle(xy(.2,.2), size*.6, size*.6,
                                       fill=False, ec=INK, lw=.9, zorder=10))
            else:
                ax.add_patch(Polygon([xy(.5,.12),xy(.87,.8),xy(.13,.8)],
                                     fill=False, ec=INK, lw=.9, zorder=10))
        elif kind == 'landmarks':
            for u,v,color in [(.3,.3,'#aa359e'),(.72,.4,'#078ba0'),(.47,.78,'#9e7a00')]:
                ax.add_patch(Circle(xy(u,v), size*.14, fc=color, ec='none', zorder=10))
        elif kind == 'embedding':
            line([(.25,.25),(.75,.25),(.75,.75),(.25,.75),(.25,.25)], lw=.5)
            for u,v in [(.25,.25),(.75,.25),(.25,.75),(.75,.75)]:
                ax.add_patch(Circle(xy(u,v), size*.12, fc=INK, ec='none', zorder=10))

    dense_axes = []
    def tile(x, y, im, title, sub='', cmap=None, norm=None, width=250, symbol=None):
        if symbol:
            icon(canvas, x+1, y-31, 18, symbol)
            text(x+26, y-22, title, 6.7, weight='bold', ha='left')
        else:
            text(x + width / 2, y - 22, title, 6.7, weight='bold')
        ax = axis(x, y, width)
        ax.imshow(im, cmap=cmap, norm=norm, interpolation='nearest')
        zoom_axes.append(ax)
        if sub:
            text(x + width / 2, y + width + 22, sub, 6.1, MUTED)
        if symbol:
            dense_axes.append(ax)
        return ax

    def bar(x, y, width, cmap, norm, ticks, labels):
        ax = axis(x, y, width, 5)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                            orientation='horizontal', ticks=ticks)
        cbar.ax.set_xticklabels(labels)
        cbar.ax.tick_params(labelsize=5.7, length=1.4, width=.45, pad=1.4)
        cbar.outline.set_linewidth(.4)

    # All four stages read left to right; the dense intermediary gets most space.
    panel('A', 'Input + targets', 'Image + annotation', 30, BLUE)
    panel('B', 'Dense model', 'RGB → 18 dense channels', 330, TEAL)
    panel('C', 'Recurrent tracer', 'One tracing step', 930, AMBER)
    panel('D', 'Arm tracing', 'Parallel tip → base', 1250, AMBER)

    # Shared display scales make the reference targets in A comparable to B.
    probability = Normalize(0, 1)
    heat_norm = PowerNorm(.35, vmin=0, vmax=1)
    heat_cmap = plt.get_cmap('magma_r').copy()
    heat_cmap.set_bad(MASK_BACKGROUND)
    support_cmap = plt.get_cmap('GnBu')
    prog = plt.get_cmap('viridis').copy(); prog.set_bad(MASK_BACKGROUND)
    landmark_colors = np.array([to_rgb(c) for c in ('#e568e5', '#28c5df', '#ffe55c')])

    # A. Manual polylines/points become geometric targets and arm-instance labels.
    # The same held-out example illustrates target construction, not training.
    tile(40, 145, rgb, 'RGB crop', width=220)
    arrow((150, 369), (150, 420), BLUE)
    text(170, 395, 'Annotate', 5.8, BLUE, ha='left')
    text(150, 434, 'Polylines + points', 6.7, weight='bold')
    ax = axis(40, 456, 220)
    zoom_axes.append(ax)
    ax.imshow(rgb)
    shapes = json.loads((ROOT / row['shapes_path']).read_text())
    instance_palette = plt.get_cmap('tab20')
    count = 0
    for shape in shapes:
        if shape.get('outside'):
            continue
        xy = np.asarray(shape['points']).reshape(-1, 2) * np.array([512/int(row['width']), 512/int(row['height'])])
        if shape['label'] == 'arm' and shape['type'] == 'polyline':
            color = instance_palette(count % 20)
            ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.6)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=.85)
            # Arrowheads at mid-arm and at the tip point from base to tip.
            arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            for frac in ANNOTATION_ARROW_FRACTIONS:
                at = frac * arc[-1]
                head = np.array([np.interp(at, arc, xy[:, 0]), np.interp(at, arc, xy[:, 1])])
                tail = np.array([np.interp(at - 8, arc, xy[:, 0]), np.interp(at - 8, arc, xy[:, 1])])
                arrow_artist = FancyArrowPatch(tail, head, arrowstyle='-|>', mutation_scale=7.5,
                                               lw=0, color=color, shrinkA=0, shrinkB=0, zorder=7)
                arrow_artist.set_path_effects([patheffects.Stroke(linewidth=1.0, foreground=INK),
                                               patheffects.Normal()])
                ax.add_patch(arrow_artist)
            count += 1
        elif shape['type'] == 'points' and shape['label'] in ('mouth','madreporite','butthole'):
            color = landmark_colors[('mouth','madreporite','butthole').index(shape['label'])]
            ax.scatter(xy[:,0], xy[:,1], s=16, marker='o', fc='none', ec=[color], lw=.8, zorder=8)
    assert count == int(row['arm_count'])
    text(150, 696, 'Arrows point base → tip', 5.7, MUTED)
    arrow((150, 708), (150, 752), TEAL)
    text(170, 726, 'Rasterize', 5.8, TEAL, ha='left')
    text(150, 767, 'Training targets', 6.7, weight='bold')
    target_mask = target[0] > .5
    target_direction = plt.get_cmap('twilight')((np.arctan2(target[3], target[2])+np.pi)/(2*np.pi))[:,:,:3]
    target_direction[~target_mask] = to_rgb(MASK_BACKGROUND)
    target_landmarks = np.clip(np.einsum('chw,cd->hwd', np.sqrt(target[5:8]), landmark_colors), 0, 1)
    instance_rgb = np.full((512,512,3), to_rgb(MASK_BACKGROUND))
    for arm_id in np.unique(instances[instances>0]):
        instance_rgb[instances==arm_id] = instance_palette((int(arm_id)-1)%20)[:3]
    instance_rgb[instances<0] = to_rgb('#b5bec4')
    target_views = [
        ('Support','support',target[0],support_cmap,probability),
        ('Centerline','centerline',np.ma.array(target[1],mask=~target_mask),heat_cmap,heat_norm),
        ('Direction','direction',target_direction,None,None),
        ('Progress','progress',np.ma.array(target[4],mask=~target_mask),prog,probability),
        ('Bases','base',target[8],heat_cmap,heat_norm),
        ('Tips','tip',target[9],heat_cmap,heat_norm),
        ('Landmarks','landmarks',target_landmarks,None,None),
        ('Arm IDs','embedding',instance_rgb,None,None),
    ]
    for i, (name, kind, im, cmap, norm) in enumerate(target_views):
        r, c = divmod(i, 2)
        x, y = 30+c*132, 805+r*130
        icon(canvas, x+1, y-21, 15, kind)
        text(x+22, y-13, name, 5.6, weight='bold', ha='left')
        target_ax = axis(x, y, 108)
        zoom_axes.append(target_ax)
        target_ax.imshow(im, cmap=cmap, norm=norm, interpolation='nearest')

    # B. RGB -> DINOv3 stage features -> FPN/heads -> actual dense predictions.
    # Feature scales are tensor schematics, not invented activation images.
    text(387, 123, 'RGB', 6.3, weight='bold')
    ax = axis(345, 145, 84)
    ax.imshow(rgb)
    text(387, 242, '512 × 512', 5.7, MUTED)
    path_arrow([(260, 255), (313, 255), (313, 187), (345, 187)], BLUE)
    arrow((436, 187), (466, 187), BLUE)
    block(466, 145, 160, 84, 'DINOv3', 'ConvNeXt-Base', title_size=7.2)
    arrow((626, 187), (667, 187), TEAL)
    # Four stacks, 8 units apart (each stack is its face plus a 5-unit back offset).
    stack_x, centres = 674, []
    for size in (46, 35, 25, 16):
        cx = stack_x + size/2
        centres.append(cx)
        y = 185-size/2
        for offset in (5, 2.5, 0):
            canvas.add_patch(Rectangle((cx-size/2+offset, y-offset), size, size,
                                      fc='#edf6f3', ec=TEAL, lw=.5, zorder=3))
        canvas.plot([cx, cx], [y+size+3, 216], color=TEAL, lw=.6)
        stack_x += size + 5 + 8
    text((674 + stack_x - 8) / 2, 123, '4 feature scales', 6.3, weight='bold')
    canvas.plot([centres[0], centres[-1]], [216, 216], color=TEAL, lw=.6)
    mid_stacks = (centres[0] + centres[-1]) / 2
    arrow((mid_stacks, 216), (mid_stacks, 246), TEAL, gap_start=False)
    block(466, 246, 374, 60, 'FPN + 5 prediction heads', '18 × 512 × 512', title_size=6.7)
    canvas.plot([595, 595, 450, 730], [306, 311, 311, 311], color=TEAL, lw=.85)
    for x in (450, 730):
        arrow((x, 311), (x, 330), TEAL, gap_start=False)

    # All 18 channels, grouped without inventing anatomical embedding IDs.
    rows = [367, 637, 907, 1177]
    dense_width = 205
    # One shared, fixed probability scale for support, centerline and endpoints.
    tile(345, rows[0], a[0], 'Arm support · 1 ch.', cmap=support_cmap, norm=probability, width=dense_width, symbol='support')
    tile(625, rows[0], a[1], 'Centerline · 1 ch.', cmap=heat_cmap, norm=heat_norm, width=dense_width, symbol='centerline')
    bar(400, rows[0]+dense_width+4, 95, support_cmap, probability, [0, 1], ['0', '1'])
    bar(680, rows[0]+dense_width+4, 95, 'magma_r', heat_norm, [0, 1], ['0', '1'])
    for x in (380, 660):
        text(x, rows[0]+dense_width+7, '$p$', 5.7, MUTED, ha='right')

    angle = np.arctan2(a[3], a[2])
    direction = plt.get_cmap('twilight')((angle + np.pi) / (2*np.pi))[:, :, :3]
    tile(345, rows[1], direction, 'Direction · 2 ch.', 'Hue: local arm direction', width=dense_width, symbol='direction')
    tile(625, rows[1], a[4], 'Progress · 1 ch.', cmap=prog, norm=probability, width=dense_width, symbol='progress')
    bar(680, rows[1]+dense_width+4, 110, 'viridis', probability, [0, 1], ['0', '1'])

    tile(345, rows[2], a[8], 'Arm bases · 1 ch.', cmap=heat_cmap, norm=heat_norm, width=dense_width, symbol='base')
    tile(625, rows[2], a[9], 'Arm tips · 1 ch.', cmap=heat_cmap, norm=heat_norm, width=dense_width, symbol='tip')

    landmarks = np.clip(np.einsum('chw,cd->hwd', np.sqrt(a[5:8]), landmark_colors), 0, 1)
    tile(345, rows[3], landmarks, 'Landmarks · 3 ch.', width=dense_width, symbol='landmarks')
    landmark_key_y = rows[3]+dense_width+24
    for x, name, kind, color in [(345, 'Mouth', 'mouth', '#aa359e'), (425, 'Madrep.', 'madreporite', '#078ba0'), (525, 'Anus', 'anus', '#9e7a00')]:
        icon(canvas, x-5, landmark_key_y-6, 13, kind)
        text(x+13, landmark_key_y, name, 5.8, color, ha='left')
    tile(625, rows[3], embedding, 'Embedding · 8 ch.', 'PCA', width=dense_width, symbol='embedding')

    # The same spatial window is marked in all B maps and sampled in C.
    for dense_ax in dense_axes:
        box = Rectangle(seed-16.5, 33, 33, fill=False, ec=SEED, lw=.75, zorder=12)
        box.set_path_effects([patheffects.Stroke(linewidth=1.65, foreground=INK),
                              patheffects.Normal()])
        dense_ax.add_patch(box)

    # C. One compact atlas contains every real dense channel at the seed. The
    # RGB branch and recurrent update are diagrams of the model interfaces.
    text(1060, 123, 'Dense channels', 6.7, weight='bold')
    dense_norm = Normalize(-4, 4)
    dense_cmap = plt.get_cmap('RdBu_r')
    atlas_cols, atlas_rows, atlas_gap, atlas_tile = 3, 6, 4, 33
    atlas_w = atlas_cols*atlas_tile + (atlas_cols-1)*atlas_gap
    atlas_h = atlas_rows*atlas_tile + (atlas_rows-1)*atlas_gap
    atlas = np.full((atlas_h, atlas_w, 3), to_rgb(MASK_BACKGROUND))
    for channel in range(18):
        row_i, col_i = divmod(channel, atlas_cols)
        x0, y0 = col_i*(atlas_tile+atlas_gap), row_i*(atlas_tile+atlas_gap)
        atlas[y0:y0+atlas_tile, x0:x0+atlas_tile] = dense_cmap(dense_norm(d['seed_stem'][channel]))[:,:,:3]
    for offset in (14, 7):
        canvas.add_patch(Rectangle((950+offset, 145-offset), 207, 207*atlas_h/atlas_w,
                                  facecolor='white', edgecolor=LINE, lw=.6, zorder=1))
    ax = axis(950, 145, 207, 207*atlas_h/atlas_w)
    ax.imshow(atlas, interpolation='nearest')
    for channel in range(18):
        row_i, col_i = divmod(channel, atlas_cols)
        icon(ax, col_i*(atlas_tile+atlas_gap)+1, row_i*(atlas_tile+atlas_gap)+1,
             10, CHANNEL_ICONS[channel], badge=True)
    bar(1015, 577, 90, 'RdBu_r', dense_norm, [-4, 0, 4], ['-4', '0', '4'])
    text(995, 580, '$z$', 5.7, MUTED, ha='right')
    text(1060, 624, '18 × 33 × 33', 6.3, MUTED)
    text(1060, 649, '+ seed similarity + x/y', 5.8, MUTED)
    arrow((1060, 664), (1060, 698), TEAL)
    block(973, 698, 174, 50, 'Dense CNN')

    # RGB is encoded once over the full crop; its local feature patches join
    # the dense features. The yellow box continues to mark dense-patch sampling.
    text(987, 774, 'RGB at tip', 6.3, weight='bold')
    # The same 36 x 36-px view of the arm tip as the next-step inset; yellow: the 33 x 33 dense window.
    ax = axis(945, 792, 84)
    ax.imshow(rgb, interpolation='lanczos')
    ax.add_patch(Rectangle(seed-16.5, 33, 33, fill=False, ec=SEED, lw=.9, zorder=5))
    ax.set_xlim(seed[0]-STEP_INSET_HALF_PX, seed[0]+STEP_INSET_HALF_PX)
    ax.set_ylim(seed[1]+STEP_INSET_HALF_PX, seed[1]-STEP_INSET_HALF_PX)
    arrow((1031, 834), (1064, 834), BLUE)
    block(1064, 804, 136, 60, 'RGB encoder', 'fine + coarse', color=BLUE, fill='#edf3f8', title_size=6.1)
    path_arrow([(973, 723), (930, 723), (930, 936), (955, 936)], TEAL)
    arrow((1132, 864), (1132, 910), BLUE)
    block(955, 910, 234, 55, 'Fusion + attention', color=AMBER, fill='#fbf5e8')
    arrow((1072, 965), (1072, 1022), AMBER)

    block(955, 1022, 234, 76, 'GRU', 'Seed + previous step', color=AMBER, fill='#fbf5e8')
    # A visible hidden-state loop marks recurrence. Its attention conditioning
    # is described in the caption rather than adding another crossing arrow.
    path_arrow([(1189, 1060), (1206, 1060), (1206, 989), (1125, 989), (1125, 1022)], AMBER)
    text(1170, 1005, '$h$', 6.8, AMBER)
    arrow((1072, 1098), (1072, 1174), AMBER)
    block(955, 1174, 234, 92, 'Next step + state', 'Direction · length', color=AMBER, fill='#fbf5e8')
    # The first predicted step at the tip, drawn at true scale on the RGB crop.
    step0, prob0 = d['displacements'][0], d['state_probabilities'][0]
    state_names = ('invalid seed', 'continue', 'base', 'lost')
    inset_half = STEP_INSET_HALF_PX
    arrow((1072, 1266), (1072, 1296), AMBER)
    ax = axis(1012, 1296, 118)
    ax.imshow(rgb, interpolation='lanczos')
    ax.add_patch(Rectangle(seed-16.5, 33, 33, fill=False, ec=SEED, lw=.9, zorder=5))
    step_arrow = FancyArrowPatch(seed, seed + step0, arrowstyle='-|>', mutation_scale=8, lw=1.1, color=TRACE,
                                 shrinkA=0, shrinkB=0, zorder=7)
    step_arrow.set_path_effects([patheffects.Stroke(linewidth=2.0, foreground='#111111'), patheffects.Normal()])
    ax.add_patch(step_arrow)
    ax.scatter(*seed, s=5, color=TRACE, edgecolor='#111111', linewidth=.4, zorder=8)
    ax.set_xlim(seed[0]-inset_half, seed[0]+inset_half)
    ax.set_ylim(seed[1]+inset_half, seed[1]-inset_half)
    best = int(np.argmax(prob0))
    text(1138, 1318, 'Next step', 5.8, MUTED, ha='left')
    text(1138, 1342, f'Δ = {np.linalg.norm(step0):.1f} px', 5.8, INK, ha='left')
    text(1138, 1366, state_names[best], 5.8, AMBER, ha='left', weight='bold')
    # As in Figure 1: a probability >= 0.995 would print a misleading "1.00".
    text(1138, 1388, 'p > 0.99' if prob0[best] >= 0.995 else f'p = {prob0[best]:.2f}', 5.8, MUTED, ha='left')

    # D. A highlighted trajectory and three synchronized batched-rollout views.
    ax = tile(1250, 145, rgb, 'Tip-to-base trace', width=320)
    ax.plot(points[:, 0], points[:, 1], color='#111111', lw=2.5)
    ax.plot(points[:, 0], points[:, 1], color=TRACE, lw=1.3)
    for i in (5, 15, 24):
        ann = ax.annotate('', xy=points[i+2], xytext=points[i],
                          arrowprops=dict(arrowstyle='-|>', color=TRACE, lw=1.0, mutation_scale=10))
        ann.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=1.8, foreground='#111111'),
                                          patheffects.Normal()])
    ax.add_patch(Rectangle(seed-16.5, 33, 33, fill=False, ec=SEED, lw=1.1))
    ax.scatter(*points[-1], s=14, marker='s', color=TRACE, edgecolor='black', linewidth=.5, zorder=8)
    for label, xy, xytext in [('Tip', seed, (418, 212)), ('Base', points[-1], (265, 325))]:
        ann = ax.annotate(label, xy=xy, xytext=xytext, fontsize=6.7, ha='center', color='white',
                          arrowprops={'arrowstyle':'-', 'color':'white', 'lw':.7})
        ann.set_path_effects([patheffects.withStroke(linewidth=2, foreground='#18242c')])
    text(1410, 491, '33 × 33 patch', 6.1, MUTED)
    text(1410, 531, f'Batch tracing · {len(batch["walkers"])} arms', 6.7, weight='bold')
    batch_snapshot_display = []
    # Start at the tips (step 0), then the recorded midpoint and final updates.
    snapshots = [{'label': 'Start', 'step': 0}] + [f for f in batch['snapshots'] if f['label'] != 'Early']
    for frame, y in zip(snapshots, (585, 865, 1145)):
        t = frame['step']
        ax = tile(1290, y, rgb, f'{frame["label"]} · step {t}', width=235)
        paths = []
        walkers = sorted(batch['walkers'], key=lambda w: w['walker_index'] == batch['selected_walker_index'])
        for walker in walkers:   # highlighted walker drawn last, on top
            rollout = walker['rollout']
            upto = min(t, len(rollout)-1)
            xy = np.asarray([r['positions'] for r in rollout[:upto+1]])
            selected = walker['walker_index'] == batch['selected_walker_index']
            finished = t >= walker['steps']
            if len(xy) > 1:
                ax.plot(xy[:,0], xy[:,1], color='#111111', lw=1.8, zorder=3)
                ax.plot(xy[:,0], xy[:,1], color=TRACE if selected else '#f7f7f7', lw=.9, zorder=4)
            # Every walker's 33 x 33 sampling window travels with its head from the tip.
            box = Rectangle(xy[-1]-16.5, 33, 33, fill=False, ec=SEED if selected else 'white',
                            lw=.8 if selected else .6, zorder=8 if selected else 6)
            box.set_path_effects([patheffects.Stroke(linewidth=1.6 if selected else 1.3, foreground=INK),
                                  patheffects.Normal()])
            ax.add_patch(box)
            paths.append({'walker_index':walker['walker_index'], 'visible_positions':len(xy),
                          'head_xy':xy[-1].tolist(), 'finished':finished})
        batch_snapshot_display.append({**frame, 'walkers':paths})
    # One bold arrow spans the three snapshots: tracing advances from start to complete.
    canvas.add_patch(FancyArrowPatch((1555, 703), (1555, 1263), arrowstyle='-|>', mutation_scale=13,
                                     color=AMBER, lw=1.6, shrinkA=0, shrinkB=0))
    text(1408, 1404, 'Spatial tracing steps', 5.8, MUTED)

    for zoom_ax in zoom_axes:
        zoom_ax.set_xlim(zoom[0], zoom[1])
        zoom_ax.set_ylim(zoom[3], zoom[2])
    base = HERE / 'figure_02_dense_outputs_and_tracing'
    fig.savefig(base.with_suffix('.pdf'), dpi=300, metadata={
        'Title': 'DeepStarNoodles dense outputs and recurrent arm tracing',
        'Creator': 'build_figure_02.py', 'Subject': 'Published-model inference on one validation crop'})
    fig.savefig(base.with_suffix('.svg'), dpi=300)
    fig.savefig(base.with_suffix('.png'), dpi=300)
    plt.close(fig)
    provenance = {
        'figure': 2, 'status': 'revised draft: D shows simultaneous batched tracing at early, midpoint and complete updates', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'inference_record': str((DATA/'inference.json').relative_to(ROOT)),
        'inference_record_sha256': digest(DATA/'inference.json'),
        'target_record': str((DATA/'targets.json').relative_to(ROOT)),
        'target_record_sha256': digest(DATA/'targets.json'),
        'dense_model_source': model_source,
        'batched_rollout_record': str((DATA/'batched_rollout.json').relative_to(ROOT)),
        'batched_rollout_record_sha256': digest(DATA/'batched_rollout.json'),
        'build_script_sha256': digest(__file__), 'size_inches': [7.48, 7.48*H/W],
        'display': {'rgb_gamma': GAMMA, 'applied_to_inference': False,
            'example_zoom_px': {'x': [zoom[0], zoom[1]], 'y': [zoom[2], zoom[3]], 'side': side,
                                'rule': 'square around annotated shapes, support>0.5, exported traces and every walker tip window, '
                                        f'+{ZOOM_MARGIN_PX} px margin; applied to every view of the example except the '
                                        'two small model-input icons in B and C, which show the full 512 x 512 input'},
            'dense_model_diagram': {'input':'Same full 512x512 RGB crop as A, internally ImageNet-normalized by the backbone.', 'backbone':'DINOv3 ConvNeXt-Base', 'stage_channels':[128,256,512,1024], 'feature_display':'Four decreasing-size stacks are schematic feature tensors, not activation images; drawn dimensions do not encode exact spatial ratios.', 'neck':'FPN: four lateral projections and top-down bilinear fusion, followed by refinement.', 'head_channels':{'support_centerline':2,'direction_progress':3,'body_landmarks':3,'bases_tips':2,'embedding':8}, 'output_shape':[18,512,512], 'output_resolution':'Five heads share the FPN feature map; concatenated head outputs are bilinearly resized to the input resolution. FPN and heads are grouped into one block.'},
            'support_mask': {'threshold': SUPPORT_THRESHOLD, 'rule': 'activated support > threshold',
                'scope': 'No B channel is display-masked. The mask is used only to fit the embedding PCA, to choose direction-arrow positions and to set the example zoom. A targets keep target-support masking.',
                'fit_pixels': int(mask.sum())},
            'support': 'sigmoid; GnBu linear 0..1; unmasked (masking support by itself hid its falloff)',
            'centerline': 'sigmoid; magma_r PowerNorm(gamma=.35), fixed 0..1, shared with endpoints; unmasked',
            'annotation_direction': 'arrowheads at 50% and 100% of each annotated polyline arc length (the second ends exactly at the tip), pointing base to tip, in the arm colour',
            'direction': 'tanh dx/dy; twilight atan2 hue at every pixel (unmasked); no arrow overlay',
            'progress': 'sigmoid; viridis 0..1; unmasked',
            'endpoint_heatmaps': 'sigmoid; magma_r PowerNorm(gamma=.35), fixed 0..1 range, shared with centerline; unmasked; no spatial dilation',
            'landmarks': 'sqrt(sigmoid) additive RGB colors for mouth, madreporite, anus; unmasked; fixed 0..1, clipped composite; no per-channel renormalization',
            'color_keys': 'Four compact 5-unit-high keys: support GnBu 0..1, centerline/endpoint magma_r 0..1 (power .35), progress 0..1, and standardized dense-channel z.',
            'embedding': 'Raw channels 10..17; PCA fit on support >.5 pixels and applied to every pixel; per-PC 2..98 percentile display; unmasked',
            'dense_patch': 'All 18 standardized channels from the observed 21-channel dense-stem input, arranged row-major in a 3-column x 6-row atlas. Shared vector icons identify channel types and match the B labels. All eight embedding channels share an embedding icon; arrow direction distinguishes dx/dy. Fixed z range -4..4; no support masking. Back plates are schematic tensor layers.',
            'dense_channel_atlas': {'array':'seed_stem[:18]', 'channel_order':list(range(18)), 'channel_icons':CHANNEL_ICONS, 'rows':atlas_rows, 'columns':atlas_cols, 'tile_pixels':atlas_tile, 'gap_pixels':atlas_gap, 'vmin':-4, 'vmax':4},
            'sampled_region': {'center_xy':seed.tolist(), 'width_pixels':33, 'height_pixels':33, 'display':'Outlined yellow boxes in all eight B maps show the same seed window as C. A single connector from the centerline map points to the C atlas. B shows activated/display-masked fields; C shows recorded standardized raw fields.'},
            'seed_similarity': 'Included as an auxiliary CNN input alongside relative x/y; its standalone heatmap is omitted.',
            'rgb_features': 'Full RGB crop with seed box -> RGB encoder -> local fine/coarse features -> fusion/attention. No RGB feature PCA is displayed; the yellow box denotes dense sampling, not an RGB receptive field.',
            'recurrent_diagram': 'Dense CNN and RGB feature branches merge through fusion/spatial attention. GRU receives 64 context + 8 seed embedding + 2 previous displacement values, with a separate 128-value hidden state. GRU loop shows hidden-state recurrence; hidden/seed/previous-step attention conditioning is omitted graphically and recorded in caption. Output heads predict displacement direction/length and trace state.',
            'annotation': 'Original arm polylines and annotated body points scaled by 512/source dimension. Polylines colored by arbitrary per-image arm ID using tab20; squares mark the first (base) point, triangles the last (tip) point. Only annotated body landmarks are drawn; this image has one anus point, no mouth/madreporite points.',
            'training_targets': 'rasterize_targets with the saved dense-checkpoint parameters, no augmentation; exact parity against HydraDenseDataset and saved RGB. Same held-out example illustrates supervision construction, not membership in the training set. Ten scalar/vector target channels plus an instance label map are shown in eight groups matching B. Support/centerline/direction/progress use target-support masking for display; landmarks and endpoints are unmasked. Same heatmap, direction-hue, progress and landmark scales as B. Arm-ID colors are tab20, matching annotated polylines; ignored overlaps gray. Arm IDs supervise learned embeddings; no fixed eight-dimensional embedding targets are invented. RGB->manual annotation->rasterization is a supervision construction path, not inference.',
            'trace': 'Observed positions and predicted displacements, including terminal position; no smoothing of highlighted rollout. '
                     f'Upper view shows one complete {len(points)-1}-update tip-to-base path and the initial seed box.',
            'batch_tracing': f'All {len(batch["walkers"])} initial walkers advance together per decoder call. Three snapshots use common global update counts ({",".join(str(f["step"]) for f in snapshots)}), not per-arm fractional progress. Raw observed tip-to-base prefixes are white; the same highlighted walker is cyan. Circles indicate current heads; squares indicate stopped heads. Finished paths remain in later snapshots. The 33px yellow patch follows the highlighted current head. No smoothing, interpolated motion or slicing of exported base-to-tip polylines is used.',
            'batch_snapshots': batch_snapshot_display,
            'export': 'All exported paths remain in retained inference data, with the original decoder postprocessing; the figure now shows raw batched rollouts rather than a separate postprocessed export view.'},
        'pca': {'embedding': embedding_pca},
        'outputs': {kind: {'path': str(base.with_suffix('.'+kind).relative_to(ROOT)),
                           'sha256': digest(base.with_suffix('.'+kind))} for kind in ('pdf','svg','png')},
    }
    base.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    # Every number printed in the figure, with its definition and source.
    rel = lambda name: str((DATA / name).relative_to(ROOT))
    spec = inference['dense_output_spec']
    assert raw.shape == (18, 512, 512) and spec['embedding']['dim'] == 8
    assert inference['trace_params']['dense_patch_size'] == 33
    assert len(batch['walkers']) == inference['exported_arm_count'] == len(traces['arms'])
    assert snapshots[-1]['step'] == len(points) - 1 == inference['selected_arm_steps']
    numbers = {
        'figure': 2, 'generated_at_utc': provenance['generated_at_utc'],
        'models': {'dense': inference['dense'],
                   'tracer': {k: inference['tracer'][k] for k in ('path', 'sha256', 'invalid_seed_threshold', 'epoch_or_cycle')}},
        'example': {'image_path': row['image_path'], 'shapes_path': row['shapes_path'], 'uid': row.get('uid'),
                    'split': 'validation', 'index': inference['selection']['index']},
        'printed': [
            {'label': '512 × 512', 'value': [512, 512], 'definition': 'side of the RGB crop given to the dense model, pixels', 'source': rel('predictions.npz') + ' (image_rgb shape)'},
            {'label': 'RGB → 18 dense channels; 18 × 512 × 512', 'value': list(raw.shape), 'definition': 'shape of the dense model output for one crop', 'source': rel('predictions.npz') + ' (raw shape)'},
            {'label': '4 feature scales', 'value': 4, 'definition': 'number of backbone stages passed to the feature pyramid', 'source': 'CORE/deepstarnoodles/model.py'},
            {'label': 'FPN + 5 prediction heads', 'value': 5, 'definition': 'number of output heads of the dense model', 'source': 'CORE/deepstarnoodles/model.py'},
            {'label': 'Arm support · 1 ch.; Centerline · 1 ch.; Direction · 2 ch.; Progress · 1 ch.; Arm bases · 1 ch.; Arm tips · 1 ch.; Landmarks · 3 ch.; Embedding · 8 ch.',
             'value': {'arm_support': 1, 'arm_centerline': 1, 'direction': 2, 'arm_progress': 1, 'arm_base': 1, 'arm_tip': 1, 'landmarks': 3, 'embedding': 8},
             'definition': 'number of output channels in each displayed group (sum 18)', 'source': rel('inference.json') + ' (dense_output_spec)'},
            {'label': '18 × 33 × 33; 33 × 33 patch', 'value': [18, 33, 33], 'definition': 'dense patch read by the tracer at each step', 'source': rel('inference.json') + ' (trace_params.dense_patch_size)'},
            {'label': f'Δ = {np.linalg.norm(step0):.1f} px', 'value': float(np.linalg.norm(step0)), 'definition': 'length of the first predicted step of the highlighted arm, 512-pixel crop coordinates', 'source': rel('predictions.npz') + ' (displacements[0])'},
            {'label': f'{state_names[best]} ' + ('p > 0.99' if prob0[best] >= 0.995 else f'p = {prob0[best]:.2f}'), 'value': float(prob0[best]), 'definition': 'largest state probability at the first step of the highlighted arm', 'source': rel('predictions.npz') + ' (state_probabilities[0])'},
            {'label': f'Batch tracing · {len(batch["walkers"])} arms', 'value': len(batch['walkers']), 'definition': 'arms traced on this crop (walkers started at tip candidates, all exported)', 'source': rel('batched_rollout.json')},
            {'label': ' / '.join(f'{f["label"]} · step {f["step"]}' for f in snapshots), 'value': [f['step'] for f in snapshots], 'definition': 'number of tracer updates applied to all arms at each displayed snapshot; the last equals the longest trace', 'source': rel('batched_rollout.json') + ' (snapshots)'},
            {'label': 'colour bars 0 to 1 and −4 to 4', 'value': [0, 1, -4, 4], 'definition': 'display ranges of probabilities and of standardised dense values', 'source': 'build_figure_02.py'},
        ],
        'not_printed': {'annotated_arms_in_example': int(row['arm_count']), 'highlighted_trace_updates': int(len(points) - 1),
                        'highlighted_trace_stop': inference['selected_arm_stop']},
    }
    (HERE / 'figure_02_numbers.json').write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + '\n')
    print('Built Figure 2 PDF, SVG, PNG and display provenance.')


if __name__ == '__main__':
    build()
