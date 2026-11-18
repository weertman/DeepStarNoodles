#!/usr/bin/env python3
"""Build Figure 1: what each DeepStarNoodles model outputs for one video frame.

Run ``prepare_figure_01.py`` first. This script needs only NumPy, OpenCV and
matplotlib, and reuses the saved model outputs for layout edits.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'deepstarnoodles-matplotlib'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import PowerNorm, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'figure_01_data'
STEM = 'figure_01_workflow'

W, H = 7.48, 3.16
INK, MUTED, LINE = '#20303c', '#5d6d79', '#c9d1d7'
BLUE, TEAL, AMBER, PLUM = '#35668f', '#237b76', '#93651e', '#6a4c93'
LAYER_BG = np.array([0.955, 0.96, 0.965])
GAMMA = 0.55
STEP_ARROW_SCALE = 2.0
ZOOM_MARGIN_PX = 6        # same rule as Figure 2: arms stop just short of the view edge
FLOW = '#56636e'
LANDMARK_COLORS = ('#e568e5', '#28c5df', '#ffe55c')   # mouth, madreporite, anus, as in Figure 2
SOURCE_WINDOW = (170, 2430, 290, 1830)   # x0, x1, y0, y1 in source pixels (tank interior)
ARM_COLORS = ['#e6194b', '#3cb44b', '#ffbf00', '#4363d8', '#f58231', '#911eb4', '#29b6c8',
             '#f032e6', '#9bc40c', '#008080', '#c77d4a', '#7f7fff', '#ff6f91', '#6d8a00']
OUTLINE = [patheffects.Stroke(linewidth=2.1, foreground='#111111'), patheffects.Normal()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gamma(rgb):
    return np.clip((np.asarray(rgb, np.float32) / 255.0) ** GAMMA, 0, 1)


# ---------------------------------------------------------------- dense layers
def blend(base, color, alpha):
    alpha = np.clip(alpha, 0, 1)[..., None]
    return base * (1 - alpha) + np.asarray(color, np.float32) * alpha


def dense_layers(act, zoom):
    """Seven display layers covering all 18 channels (top first), unmasked, as in Figure 2."""
    act = act.astype(np.float32)
    s_, c, dx, dy, prog = act[0], act[1], act[2], act[3], act[4]
    base, tip, emb = act[8], act[9], act[10:18]
    heat = PowerNorm(0.35, vmin=0, vmax=1)
    support = plt.get_cmap('GnBu')(np.clip(s_, 0, 1))[..., :3]
    centerline = plt.get_cmap('magma_r')(heat(np.clip(c, 0, 1)))[..., :3]
    direction = plt.get_cmap('twilight')((np.arctan2(dy, dx) + np.pi) / (2 * np.pi))[..., :3]
    progress = plt.get_cmap('viridis')(np.clip(prog, 0, 1))[..., :3]
    landmarks = np.clip(np.einsum('chw,cd->hwd', np.sqrt(np.clip(act[5:8], 0, 1)),
                                  np.array([to_rgb(x) for x in LANDMARK_COLORS])), 0, 1)
    ends = plt.get_cmap('magma_r')(heat(np.clip(np.maximum(base, tip), 0, 1)))[..., :3]

    pixels = emb.reshape(8, -1).T
    keep = s_.reshape(-1) > 0.5
    centre = pixels[keep].mean(0)
    _u, _sv, vt = np.linalg.svd(pixels[keep] - centre, full_matrices=False)
    axes = vt[:3] * np.sign(vt[:3][np.arange(3), np.abs(vt[:3]).argmax(1)])[:, None]
    proj = (pixels - centre) @ axes.T
    lo, hi = np.percentile(proj[keep], 2, 0), np.percentile(proj[keep], 98, 0)
    embedding = np.clip((proj - lo) / (hi - lo), 0, 1).reshape(s_.shape + (3,))
    pca_record = {'centre': centre.tolist(), 'axes': axes.tolist(), 'p2': lo.tolist(), 'p98': hi.tolist(),
                  'fit_pixels': int(keep.sum())}
    zx0, zx1, zy0, zy1 = (int(round(v)) for v in zoom)
    view = lambda img: img[max(zy0, 0):zy1, max(zx0, 0):zx1]
    layers = [
        ('Support · 1 ch', view(support)),
        ('Centerline · 1 ch', view(centerline)),
        ('Direction · 2 ch', view(direction)),
        ('Progress · 1 ch', view(progress)),
        ('Landmarks · 3 ch', view(landmarks)),
        ('Base · tip · 2 ch', view(ends)),
        ('Embedding · 8 ch', view(embedding)),
    ]
    return layers, pca_record


def oblique(rgb, out_w, out_h, skew):
    """Warp a square image onto a parallelogram (RGBA, transparent outside)."""
    face_w = out_w - skew
    small = cv2.resize(rgb.astype(np.float32), (face_w, out_h), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [face_w, 0], [0, out_h]])
    dst = np.float32([[skew, 0], [out_w, 0], [0, out_h]])
    matrix = cv2.getAffineTransform(src, dst)
    rgba = np.dstack([small, np.ones(small.shape[:2], np.float32)])
    warped = cv2.warpAffine(rgba, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(*LAYER_BG.tolist(), 0.0))
    alpha = warped[..., 3:4]
    warped[..., :3] = np.where(alpha > 1e-3, warped[..., :3], LAYER_BG)
    return np.clip(warped, 0, 1)


# ---------------------------------------------------------------- main build
def build():
    hero = np.load(DATA / 'hero.npz')
    tracks = json.loads((DATA / 'tracks.json').read_text())
    selection = json.loads((DATA / 'selection.json').read_text())
    inference = json.loads((DATA / 'inference.json').read_text())
    hero_index = selection['hero_frame']
    record = tracks['frames'][hero_index]
    hero_arms = record['arms']
    rollout = json.loads((DATA / 'rollout.json').read_text())
    walkers = rollout['walkers']
    focus = walkers[rollout['highlight_walker_index']]

    # Shared example window (Figure 2 rule): support > 0.5, exported arms and every walker's
    # starting 33-px window, plus a small margin.
    act = hero['activated']
    half = rollout['dense_patch_size'] / 2
    sy_, sx_ = np.nonzero(act[0] > 0.5)
    pts = [np.column_stack([sx_, sy_])] + [np.asarray(a_['polyline_crop']) for a_ in hero_arms]
    pts += [np.asarray(w['rollout'][0]['positions'])[None] + np.array([[-half, -half], [half, half]]) for w in walkers]
    lo_, hi_ = np.min([q.min(0) for q in pts], 0), np.max([q.max(0) for q in pts], 0)
    side = float((hi_ - lo_).max() + 2 * ZOOM_MARGIN_PX)
    zx0, zy0 = np.clip((lo_ + hi_) / 2 - side / 2, 0, 512 - side)
    zoom = (float(zx0), float(zx0 + side), float(zy0), float(zy0 + side))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'deepstarnoodles-figure-01-v4'})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def axes_in(x, y, w, h, **kw):
        return fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H), **kw)

    ov = axes_in(0, 0, W, H, zorder=20)
    ov.set(xlim=(0, W), ylim=(H, 0))
    ov.axis('off')
    ov.patch.set_alpha(0)

    def text(x, y, s, size=7, color=INK, weight='normal', ha='left', va='center', **kw):
        return ov.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)

    def heading(x, y, letter, title, color):
        text(x, y, letter, size=10.5, weight='bold')
        text(x + 0.2, y - .005, title, size=7.4, color=INK)   # plain ink; colour only encodes data

    def arrow(x0, y0, x1, y1, color=FLOW, lw=1.0, scale=10):
        ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.45,head_width=0.22',
                                     mutation_scale=scale, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                     joinstyle='miter', capstyle='butt'))

    def image_axes(x, y, w, h, edge=LINE, lw=0.75):
        ax = axes_in(x, y, w, h)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(edge); spine.set_linewidth(lw)
        return ax

    row_top = 0.64
    st_x, c_x = 3.25, 5.62
    heading(0.08, 0.14, 'A', 'Detect and crop the animal', BLUE)
    heading(st_x, 0.14, 'B', 'Predict dense geometric fields', TEAL)
    heading(c_x, 0.14, 'C', 'Trace arms step by step', AMBER)
    text(0.08, 0.36, f"StarSeg detector · {inference['checkpoints']['starseg']['architecture'].split(' (')[0]}",
         color=MUTED, size=6.6)
    text(st_x, 0.36, 'Dense field model · DINOv3 ConvNeXt-B + FPN', color=MUTED, size=6.6)
    text(c_x, 0.36, 'Recurrent arm tracer (GRU)', color=MUTED, size=6.6)

    # A: video frame with StarSeg mask, box and square crop window, then the full model crop
    x0, x1, y0, y1 = SOURCE_WINDOW
    cb_x, cb_w, cb_y = 1.68, 1.36, row_top
    fa_x, fa_w = 0.08, 1.5
    fa_h = fa_w * (y1 - y0) / (x1 - x0)
    fa_y = cb_y + (cb_w - fa_h) / 2
    label_y = cb_y + cb_w + 0.13
    flow_y = cb_y + cb_w / 2
    ax = image_axes(fa_x, fa_y, fa_w, fa_h)
    ax.imshow(gamma(hero['frame_rgb_half']), extent=(0, 2600, 2128, 0), interpolation='lanczos')
    ax.add_patch(Polygon(hero['mask_polygon'], closed=True, facecolor=(0.35, 0.75, 1.0, 0.35), edgecolor='#7cc8ff', lw=0.5))
    bx0, by0, bx1, by1 = hero['bbox_xyxy']
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec='#ffd54f', lw=0.6))
    cl, ct, cs = hero['crop_left_top_side']
    ax.add_patch(Rectangle((cl, ct), cs, cs, fill=False, ec='white', lw=0.8, ls=(0, (3, 2))))
    ax.set(xlim=(x0, x1), ylim=(y1, y0))
    text(fa_x + fa_w / 2, label_y, 'Video frame · 2600 × 2128 px', size=6.3, color=MUTED, ha='center')

    axc = image_axes(cb_x, cb_y, cb_w, cb_w, edge=BLUE)
    axc.imshow(gamma(hero['model_input_rgb']), extent=(0, 512, 512, 0), interpolation='lanczos')
    sx = lambda v: fa_x + (v - x0) / (x1 - x0) * fa_w
    sy = lambda v: fa_y + (v - y0) / (y1 - y0) * fa_h
    for (px, py), (qx, qy) in (((cl + cs, ct), (cb_x, cb_y)), ((cl + cs, ct + cs), (cb_x, cb_y + cb_w))):
        ov.plot([sx(px), qx], [sy(py), qy], color='#9fb3c4', lw=0.5, ls=(0, (2, 1.5)))
    text(cb_x + cb_w / 2, label_y, 'RGB crop · 512 × 512', size=6.3, color=MUTED, ha='center')

    # B: exploded stack of all 18 activated channels, zoomed to the star
    layers, pca_record = dense_layers(act, zoom)
    n_layers = len(layers)
    st_w, st_h, st_skew = 1.34, 0.40, 0.40
    stack_top = row_top
    st_gap = (2.08 - st_h) / (n_layers - 1)
    dpi_px = 560
    for i, (name, rgb) in reversed(list(enumerate(layers))):
        ly = stack_top + i * st_gap
        img = oblique(rgb, int(st_w * dpi_px), int(st_h * dpi_px), int(st_skew * dpi_px))
        la = axes_in(st_x, ly, st_w, st_h)
        la.imshow(img, extent=(0, st_w, st_h, 0), interpolation='lanczos')
        la.set(xlim=(0, st_w), ylim=(st_h, 0)); la.axis('off')
        la.add_patch(Polygon([[st_skew, 0], [st_w, 0], [st_w - st_skew, st_h], [0, st_h]], closed=True,
                             fill=False, ec='#8fa0ab', lw=0.5))
        la.set_zorder(5 + (n_layers - i))
        text(st_x + st_w + 0.05, ly + st_h * 0.5, name, size=6.3)
    labels_right = st_x + st_w + 0.05 + 0.92

    # C: recurrent tracing on the same crop (same zoom), then the unrolled steps of one arm
    c_w = W - 0.06 - c_x
    cc_w = cb_w
    cc_x, cc_y = c_x + (c_w - cc_w) / 2, row_top
    axt = image_axes(cc_x, cc_y, cc_w, cc_w, edge=AMBER)
    axt.imshow(gamma(hero['model_input_rgb']) * 0.8, extent=(0, 512, 512, 0), interpolation='lanczos')
    for k, arm in enumerate(hero_arms):
        p_ = np.asarray(arm['polyline_crop'])
        axt.plot(p_[:, 0], p_[:, 1], color=ARM_COLORS[k % len(ARM_COLORS)], lw=1.0, alpha=0.9, solid_capstyle='round',
                 path_effects=[patheffects.Stroke(linewidth=1.8, foreground='#111111'), patheffects.Normal()])
    for w in walkers:
        if w is focus:
            continue
        q = np.asarray([r['positions'] for r in w['rollout']])
        axt.plot(q[:, 0], q[:, 1], ls='none', marker='o', ms=1.0, mfc='white', mec='none', alpha=0.85)
    fq = np.asarray([r['positions'] for r in focus['rollout']])
    n_steps = len(fq) - 1
    chain = sorted(set(int(round(v)) for v in np.linspace(0, n_steps, 5)))
    for t in chain:   # the true-scale 33 x 33 window read at each unrolled step
        axt.add_patch(Rectangle((fq[t, 0] - half, fq[t, 1] - half), 2 * half, 2 * half, fill=False,
                                ec='#ffd54f', lw=0.7, joinstyle='miter'))
    axt.plot(fq[:, 0], fq[:, 1], color='#00e5ff', lw=0.9, marker='o', ms=1.8, mfc='#00e5ff', mec='#111111', mew=0.3)
    axt.set(xlim=(zoom[0], zoom[1]), ylim=(zoom[3], zoom[2]))

    # unrolled recurrence for the highlighted arm: field window -> GRU -> step; states grouped below
    support_view = plt.get_cmap('GnBu')(np.clip(act[0], 0, 1))[..., :3]
    th = 0.28
    tg = (c_w - 5 * th) / 4
    th_y = cc_y + cc_w + 0.21
    g_y, g_h = th_y + th + 0.09, 0.15
    names = rollout['state_names']
    for k, t in enumerate(chain):
        tx = c_x + k * (th + tg)
        cx_, cy_ = fq[t]
        axp = image_axes(tx, th_y, th, th, edge='#e0b000')
        axp.imshow(support_view, extent=(0, 512, 512, 0), interpolation='lanczos')
        d = focus['rollout'][t]['displacement']
        axp.annotate('', xy=(cx_ + STEP_ARROW_SCALE * d[0], cy_ + STEP_ARROW_SCALE * d[1]), xytext=(cx_, cy_),
                     arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.25', color='#00e5ff', lw=0.9,
                                     mutation_scale=6, shrinkA=0, shrinkB=0,
                                     path_effects=[patheffects.Stroke(linewidth=1.5, foreground='#16303a'),
                                                   patheffects.Normal()]))
        axp.plot(cx_, cy_, marker='o', ms=1.8, mfc='#00e5ff', mec='#16303a', mew=0.4)
        axp.set(xlim=(cx_ - half, cx_ + half), ylim=(cy_ + half, cy_ - half))
        label = 'tip' if t == 0 else ('base' if t == n_steps else f'step {t}')
        text(tx + th / 2, th_y - 0.08, label, size=6.0, color=MUTED, ha='center')
        ov.add_patch(FancyBboxPatch((tx, g_y), th, g_h, boxstyle='round,pad=0,rounding_size=0.03',
                                    fc='#fbf5e8', ec=AMBER, lw=0.6))
        text(tx + th / 2, g_y + g_h / 2, 'GRU', size=6.0, color=AMBER, weight='bold', ha='center')
        arrow(tx + th / 2, th_y + th + 0.01, tx + th / 2, g_y - 0.005, color=AMBER, lw=0.7, scale=6)
        if k:
            arrow(tx - tg + 0.01, g_y + g_h / 2, tx - 0.01, g_y + g_h / 2, color=AMBER, lw=0.7, scale=6)
            text(tx - tg / 2, g_y + g_h / 2 - 0.075, 'h', size=6.0, color=AMBER, ha='center', style='italic')
    states, printed_states = [], []
    for k, t in enumerate(chain):
        prob = focus['rollout'][t]['probability']
        best = int(np.argmax(prob))
        if states and states[-1]['name'] == names[best]:
            states[-1]['cells'].append(k); states[-1]['p'].append(prob[best])
        else:
            states.append({'name': names[best], 'cells': [k], 'p': [prob[best]]})
    s_y = g_y + g_h + 0.08
    for group in states:
        left = c_x + group['cells'][0] * (th + tg) + 0.02
        right = c_x + group['cells'][-1] * (th + tg) + th - 0.02
        colour_state = AMBER if group['name'] == 'base' else INK
        ov.plot([left, left, right, right], [s_y - 0.03, s_y, s_y, s_y - 0.03], color=colour_state, lw=0.6)
        lo_p, hi_p = min(group['p']), max(group['p'])
        lo_s, hi_s = f'{lo_p:.2f}', f'{hi_p:.2f}'
        # Two decimals; a group whose every step is >= 0.995 would print a misleading "1.00".
        prob_text = 'p > 0.99' if lo_p >= 0.995 else (f'p = {lo_s}' if lo_s == hi_s else f'p = {lo_s}–{hi_s}')
        text((left + right) / 2, s_y + 0.1, group['name'], size=6.0, color=colour_state, ha='center',
             weight='bold' if group['name'] == 'base' else 'normal')
        text((left + right) / 2, s_y + 0.21, prob_text, size=5.8, color=MUTED, ha='center')
        printed_states.append({'state': group['name'], 'printed': prob_text,
                               'steps': [chain[k] for k in group['cells']],
                               'probabilities': [float(v) for v in group['p']]})
    assert s_y + 0.27 <= H, f'content ends at {s_y + 0.27:.2f} in; raise H'

    # flow arrows: crop -> fields -> tracer, one style, one height
    arrow(cb_x + cb_w + 0.05, flow_y, st_x - 0.04, flow_y)   # stops at the stack's left corners
    arrow(labels_right + 0.02, flow_y, cc_x - 0.05, flow_y)

    for ext in ('pdf', 'svg', 'png'):
        kwargs = {'dpi': 300} if ext == 'png' else {'dpi': 600}
        if ext == 'pdf':
            kwargs['metadata'] = {'Title': 'DeepStarNoodles Figure 1: model outputs for one video frame',
                                  'CreationDate': None}
        elif ext == 'svg':
            kwargs['metadata'] = {'Date': None}
        fig.savefig(HERE / f'{STEM}.{ext}', **kwargs)
    plt.close(fig)

    provenance = {
        'figure': STEM,
        'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'build_script_sha256': digest(__file__),
        'inputs': {p.name: digest(p) for p in sorted(DATA.iterdir())},
        'selection': selection,
        'checkpoints': inference['checkpoints'],
        'inference_device': inference['device'], 'hydra_amp': inference['hydra_amp'],
        'hero_parity': inference['hero_parity'],
        'display': {
            'rgb_gamma': GAMMA,
            'source_window_px': SOURCE_WINDOW,
            'panel_A': 'StarSeg mask polygon (cyan), detection box (yellow) and 2x square crop window (white dashed) from the frame-87 re-run.',
            'example_zoom_px': {'x': [zoom[0], zoom[1]], 'y': [zoom[2], zoom[3]], 'side': side,
                                'rule': 'square around support>0.5, exported arms and every walker start window, '
                                        f'+{ZOOM_MARGIN_PX} px; applied to B layers and the C image (A shows the full crop)'},
            'panel_B_layers': [name for name, _ in layers],
            'panel_B_rules': {
                'all': 'unmasked activated outputs; colour scales match Figure 2',
                'support': 'GnBu linear 0..1',
                'centerline': 'magma_r PowerNorm(0.35) 0..1',
                'direction': 'twilight hue = atan2(dy, dx)',
                'progress': 'viridis 0..1',
                'landmarks': 'sqrt(probability) additive RGB: mouth #e568e5, madreporite #28c5df, anus #ffe55c',
                'base_tip': 'magma_r PowerNorm(0.35) of max(base, tip)',
                'embedding': 'first three PCs fitted on support>0.5 pixels, applied to every pixel, 2nd-98th percentile to RGB',
                'geometry': 'oblique affine projection of each zoomed layer; schematic stacking, not a 3-D measurement',
                'pca': pca_record,
            },
            'panel_C': 'exported polylines for frame 87 (colours follow export order and only distinguish arms); white dots: '
                       'recorded positions of every recurrent walker; cyan: highlighted walker; yellow boxes: true-scale 33 x 33 '
                       'windows at the five unrolled steps. Thumbnails show the support layer (GnBu) in each window with the '
                       'predicted displacement at STEP_ARROW_SCALE x length; brackets group steps by most probable recorded state.',
            'panel_C_unrolled_steps': chain,
            'panel_C_step_arrow_scale': STEP_ARROW_SCALE,
        },
        'outputs': {f'{STEM}.{ext}': digest(HERE / f'{STEM}.{ext}') for ext in ('pdf', 'svg', 'png')},
    }
    (HERE / f'{STEM}.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    # Every number printed in the figure, with its definition and source.
    frame_h, frame_w = (2 * v for v in hero['frame_rgb_half'].shape[:2])
    assert (frame_w, frame_h) == (2600, 2128) and inference['display_scale'] == 0.5
    assert hero['model_input_rgb'].shape[:2] == (512, 512) and act.shape == (18, 512, 512)
    assert sum(int(name.split(' · ')[-1].split()[0]) for name, _ in layers) == 18
    data_dir = f'{HERE.name}/figure_01_data'
    numbers = {
        'figure': STEM,
        'models': {'detector': inference['checkpoints']['starseg'], 'dense': inference['checkpoints']['dense'],
                   'tracer': {k: inference['checkpoints']['tracer'][k]
                              for k in ('path', 'sha256', 'cycle', 'seed', 'invalid_seed_threshold')}},
        'video': inference['video'],
        'printed': [
            {'label': 'StarSeg detector · YOLO26-L instance segmentation', 'value': 'YOLO26-L',
             'definition': 'Architecture read from the published detector checkpoint (yolo26l-seg.yaml).',
             'source': f'{data_dir}/inference.json: checkpoints.starseg.architecture'},
            {'label': 'Video frame · 2600 × 2128 px', 'value': [frame_w, frame_h],
             'definition': 'Width and height of decoded frame 87 of the Sleepy Stars segment.',
             'source': f'{data_dir}/hero.npz: frame_rgb_half (half-size copy) and the video file'},
            {'label': 'RGB crop · 512 × 512', 'value': [512, 512],
             'definition': 'Side of the dense-model input; the detector box is padded to a square of twice its '
                           'longer side and resized.',
             'source': f'{data_dir}/hero.npz: model_input_rgb; crop_left_top_side = '
                       f"{[float(v) for v in hero['crop_left_top_side']]} source px"},
            *[{'label': name, 'value': int(name.split(' · ')[-1].split()[0]),
               'definition': 'Number of dense-model output maps shown in this layer (18 in total).',
               'source': f'{data_dir}/hero.npz: activated (18 x 512 x 512)'} for name, _ in layers],
            {'label': 'tip, ' + ', '.join(f'step {t}' for t in chain[1:-1]) + ', base', 'value': chain,
             'definition': 'Recurrent steps of the highlighted arm shown as thumbnails: five evenly spaced steps '
                           f'from the tip seed (step 0) to the last step ({n_steps}), where the walker stopped '
                           'in state base.',
             'source': f'{data_dir}/rollout.json: walker {focus["walker_index"]}'},
            *[{'label': f"{g['state']} · {g['printed']}", 'value': g['probabilities'], 'steps': g['steps'],
               'definition': 'Softmax probability of the most probable trace state at the shown steps '
                             '("p > 0.99" when every shown step is at least 0.995; otherwise two decimals).',
               'source': f'{data_dir}/rollout.json: walker {focus["walker_index"]} probability'}
              for g in printed_states],
        ],
        'not_printed': {'arms_traced_in_frame': len(hero_arms), 'walkers': len(walkers),
                        'highlighted_arm_steps': n_steps,
                        'detection_confidence': inference['hero_detection_confidence'],
                        'window_px': int(rollout['dense_patch_size'])},
    }
    (HERE / 'figure_01_numbers.json').write_text(json.dumps(numbers, indent=2) + '\n')
    print('wrote', STEM)


if __name__ == '__main__':
    build()
