"""Edge cases: empty frames, 1- and 2-point arms, duplicate points, zero-length arms, single-arm frames,
frames with fewer than 4 arms, missing track_id and NaN origins. Public API == _reference on all of them."""
from __future__ import annotations



import numpy as np
import pytest

from noodle_polylines import PolylineContainer, compute_all, from_polylines
from noodle_polylines import _reference as R
from noodle_polylines import frames as FR, geometry as G, kinematics as KI, tracking as TR

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


def close(a, b, rtol=1e-9):
    """The closed-form Kabsch default: NaN/inf positions exact, finite values within the gate's rtol 1e-9."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert np.array_equal(fa, fb) and np.array_equal(a[~fa], b[~fb], equal_nan=True)
    assert np.all(np.abs(a[fa] - b[fb]) <= rtol * np.abs(b[fb])), np.max(np.abs(a[fa] - b[fb]) / np.abs(b[fb]))


def make(seed: int = 0, n_frames: int = 40) -> PolylineContainer:
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_frames):
        n = int(rng.choice([0, 1, 2, 3, 4, 5, 6], p=[.1, .15, .1, .15, .2, .15, .15]))
        arms = []
        for i in range(n):
            ang = 2 * np.pi * i / max(n, 1) + rng.normal(0, .1) + f * 0.01
            kind = rng.choice(['normal', 'one', 'two', 'dup', 'zero'], p=[.6, .1, .1, .1, .1])
            m = {'normal': int(rng.integers(4, 20)), 'one': 1, 'two': 2, 'dup': 8, 'zero': 5}[kind]
            r = np.arange(m) * 4.0 + 30
            pts = np.column_stack([256 + r * np.cos(ang), 256 + r * np.sin(ang)]) + rng.normal(0, .5, (m, 2))
            if kind == 'dup':
                pts[3] = pts[2]; pts[4] = pts[2]
            if kind == 'zero':
                pts[:] = pts[0]
            arms.append(pts[::-1])   # base (outer end) -> tip
        frames.append(arms)
    c = from_polylines(frames)
    F, A = c.n_frames, c.n_arms
    origin = 256 + rng.normal(0, 1, (F, 2))
    origin[7] = np.nan                                     # untraced origin on a frame that may have arms
    has_star = c.arms_per_frame() > 0
    has_star[[3, 4]] = True                                # star frames without arms
    t = np.arange(F) * 0.5
    return c.with_fields(
        time_s=t, crop=np.column_stack([rng.uniform(100, 200, F), rng.uniform(50, 150, F), rng.uniform(300, 600, F)]),
        body_origin_model=origin, support_area_model_px=rng.integers(0, 9000, F).astype(np.int32),
        support_centroid_model=256 + rng.normal(0, 1, (F, 2)), has_star=has_star,
        identity=rng.normal(0, 1, (A, 8)).astype(np.float32), meta={'fps': 2.0}).validate()


@pytest.fixture(scope='module', params=[0, 1, 2])
def c(request):
    return make(request.param)


def test_per_arm_kernels_equal_reference(c):
    pts, ao = c.points, c.arm_point_offsets
    base = G.resample_base(pts, ao)
    tip, L = G.resample_tip(pts, ao)
    Lz, kz = G.arc_length_curvature(pts, ao, 'zero')
    Ln, kn = G.arc_length_curvature(pts, ao, 'nan')
    for a in range(c.n_arms):
        p = c.arm(a)
        same(base[a], R.resample_arm(p))
        t, l_ = R.tip_anchored_resample(p, 12)
        same(tip[a], t); assert L[a] == l_
        z = R.poly_metrics(p); n = R.arm_metrics(p)
        same([Lz[a], kz[a]], [z[0], z[1]]); same([Ln[a], kn[a]], list(n))
        if len(p) == 1:
            assert Lz[a] == 0 and np.isnan(kz[a]) and np.isnan(kn[a]) and L[a] == 0
            assert np.all(base[a] == p[0]) and np.all(tip[a] == p[0])
        if len(p) == 2:
            assert kz[a] == 0.0 and np.isnan(kn[a])
        if len(p) > 1 and np.all(p == p[0]):              # zero-length arm
            assert Lz[a] == 0 and L[a] == 0 and np.all(base[a] == p[0])


def test_frames_equal_reference_fig5(c):
    r0 = 40.0
    ref = R.fig5_window(R.fig5_frames(c), onset=10.0, r0=r0, area0=1.0)
    ids = np.concatenate([np.asarray(x, np.int64) for x in ref['labels']]) if c.n_arms else np.zeros(0, np.int64)
    got_ids = TR.relink(c)
    same(got_ids[got_ids >= 0], ids)
    assert np.all(got_ids[c.arm_frame() == 7] == -1)       # NaN origin: tracker not stepped
    same(FR.fold_asymmetry(c, r0, 'fig5'), ref['rows']['fold_asymmetry'])
    same(FR.arm_length_spread(c, r0, 'fig5'), ref['rows']['arm_length_spread'])
    same(FR.arm_curvature(c, r0, 'fig5'), ref['rows']['arm_curvature'])
    same(KI.tip_motion(c, r0, got_ids), ref['rows']['tip_motion'])
    same(KI.arm_deformation(c, r0, got_ids, kabsch='svd'), ref['rows']['arm_deformation'])
    close(KI.arm_deformation(c, r0, got_ids), ref['rows']['arm_deformation'])
    n = c.arms_per_frame()
    spread = FR.arm_length_spread(c, r0, 'fig5')
    assert np.all(np.isnan(spread[n < 4]))
    assert np.all(np.isnan(FR.tip_isotropy(c)[n < 4]))
    fold = FR.fold_asymmetry(c, r0, 'fig5')
    assert np.all(np.isnan(fold[n == 0])) and np.isnan(fold[7])


def test_frames_equal_reference_fig4e(c):
    # the Fig 4 scripts never see a NaN origin on a frame with arms (a NaN polyline would break the assignment)
    origin = c.body_origin_model.copy()
    origin[7] = 256.0
    c = c.with_fields(body_origin_model=origin)
    rows, arms, polys = R.flip_tables(c)
    ids = TR.relink(c)
    track_of = {a: int(i) for a, i in enumerate(ids)}
    t = c.times()
    ev = {'flip_start_s': float(t[10]), 'upside_down_start_s': float(t[12]), 'right_side_up_start_s': float(t[30]),
          'first_time_s': float(t[0]), 'last_time_s': float(t[-1])}
    star = c.star_mask() & (c.support_area_model_px > 0)
    if not np.any(star[:10]):
        pytest.skip('no pre-onset frames in this draw')
    with np.errstate(all='ignore'):
        ref = R.fig4e_event(rows, arms, polys, track_of, ev)
    r0 = ref['r0']
    same(FR.fold_asymmetry(c, r0, 'fig4e'), ref['columns']['fold_asymmetry'])
    same(FR.arm_length_spread(c, r0, 'fig4e'), ref['columns']['arm_length_spread'])
    same(FR.arm_curvature(c, r0, 'fig4e'), ref['columns']['arm_curvature'])
    same(KI.arm_deformation(c, r0, ids, lag='prev_frame_reset', kabsch='svd'), ref['columns']['arm_deformation'])
    close(KI.arm_deformation(c, r0, ids, lag='prev_frame_reset'), ref['columns']['arm_deformation'])


def test_relink_reference_and_diagnostics(c):
    seq = {'name': 's', 'fps': 2.0, 'frames': R.stats_frames(c, frames=np.arange(c.n_frames), origin_float32=False)}
    ok = np.all(np.isfinite(c.body_origin_model), axis=1)
    pos = np.flatnonzero(ok)
    seq['frames'] = [seq['frames'][k] for k in pos]
    ref = R.analyse(seq, variants={'fused': R.RELEASED})
    d = TR.diagnostics(c, frames=pos, variants={'fused': TR.RELEASED})
    fo = c.frame_arm_offsets
    for key, ids in d['runs'].items():
        sub = pos[::int(key.rsplit('_g', 1)[1])] if '_g' in key else pos
        got = np.concatenate([ids[fo[f]:fo[f + 1]] for f in sub])
        same(got, np.concatenate([np.asarray(x, np.int64) for x in ref['runs'][key]] + [np.zeros(0, np.int64)]))
    assert d['fused']['n_ids'] == ref['fused']['n_ids']
    assert d['strides'] == ref['strides']


def test_missing_track_id_and_compute_all(c):
    assert c.track_id is None
    with pytest.raises(ValueError):
        KI.tip_motion(c, 40.0)
    out = compute_all(c, 'fig5', r0=40.0)
    assert str(out['track_id_source']) == 'relink'
    stored = c.with_fields(track_id=out['track_id'])
    with pytest.raises(ValueError):            # -1 entries (NaN-origin frame) are not a complete stored ID set
        TR.track_ids(stored, mode='stored')
    out2 = compute_all(stored, 'fig5', r0=40.0, track='relink')
    same(out2['arm_deformation'], out['arm_deformation'])


def test_single_arm_and_empty_container():
    c = from_polylines([[np.array([[0, 0], [4, 0], [8, 0]])], [], [np.array([[1, 1]])]],
                       time_s=[0.0, 1.0, 2.0], crop=np.tile([0, 0, 512.0], (3, 1)),
                       body_origin_model=np.zeros((3, 2)), identity=np.zeros((2, 8)))
    assert list(c.arms_per_frame()) == [1, 0, 1]
    fold = FR.fold_asymmetry(c, 1.0, 'fig5')
    assert fold[0] == pytest.approx(8.0) and np.isnan(fold[1]) and fold[2] == pytest.approx(np.sqrt(2))
    assert np.all(np.isnan(FR.arm_length_spread(c, 1.0, 'fig5')))
    ids = TR.relink(c)
    assert list(ids) == [0, 1] or list(ids) == [0, 0]
    empty = from_polylines([[], []], time_s=[0.0, 1.0], crop=np.tile([0, 0, 512.0], (2, 1)),
                           body_origin_model=np.zeros((2, 2)), identity=np.zeros((0, 8)))
    assert empty.n_arms == 0 and list(TR.relink(empty)) == []
    assert np.all(np.isnan(FR.fold_asymmetry(empty, 1.0, 'fig5')))
    assert G.resample_base(empty.points, empty.arm_point_offsets).shape == (0, 12, 2)


def test_nan_origin_tip_polar(c):
    ang, rad = G.tip_polar(c.points, c.arm_point_offsets, c.frame_arm_offsets, c.body_origin_model)
    nan_arms = c.arm_frame() == 7
    assert np.all(np.isnan(ang[nan_arms])) and np.all(np.isfinite(ang[~nan_arms]))
