"""Conservative output parity checks for execution optimizations.

These compare outputs to the pinned eager runtime, not to biological ground
truth. A failed gate never authorizes relaxing thresholds after seeing results.
"""
from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_TOLERANCES = {"bbox_pixels": 1.0, "polyline_crop_pixels": 1.0,
                      "identity_l2": 0.01, "confidence": 0.01,
                      "source_mapping_pixels": 0.01, "length_px": 1.0,
                      "mask_iou": 0.99, "require_same_steps": True,
                      "require_same_stop_reason": True,
                      # Schema-3 fields, compared only when both records carry them.
                      "body_origin_pixels": 1.0, "support_centroid_pixels": 1.0,
                      "support_area_fraction": 0.01, "require_same_body_origin_code": True,
                      "time_s": 1e-9,
                      # Schema-4 landmarks: a position is compared only where both runs see the landmark.
                      "landmark_probability": 0.01, "landmark_pixels": 1.0, "landmark_visible_p": 0.5}


def mask_iou(left, right, source_shape):
    if left is None or right is None:
        return 1.0 if left is None and right is None else 0.0
    # Fixed raster is sufficient for a screening gate; retain polygons for a
    # later full-resolution task-specific evaluation. Do not imply mask AP.
    h, w = source_shape
    a, b = np.zeros((512, 512), np.uint8), np.zeros((512, 512), np.uint8)
    scale = np.array([511 / max(w - 1, 1), 511 / max(h - 1, 1)])
    for polygon, mask in ((left, a), (right, b)):
        polygon = np.asarray(polygon)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise ValueError("invalid mask polygon")
        points = np.rint(polygon * scale).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def compare_body(sa, sb, frame, tolerance, issue, numeric):
    """Compare the schema-3 and -4 star fields present in both records; returns how many were compared."""
    compared = 0
    if "body_origin_code" in sa and "body_origin_code" in sb:
        compared += 1
        if tolerance["require_same_body_origin_code"] and sa["body_origin_code"] != sb["body_origin_code"]:
            issue("body_origin_code", frame)
    if "body_origin_model" in sa and "body_origin_model" in sb:
        compared += 1
        numeric("body_origin_pixels", float(np.linalg.norm(np.subtract(sa["body_origin_model"],
                                                                       sb["body_origin_model"]))), frame)
    if "support_area_model_px" in sa and "support_area_model_px" in sb:
        compared += 1
        left, right = sa["support_area_model_px"], sb["support_area_model_px"]
        numeric("support_area_fraction", abs(left - right) / max(left, right, 1), frame)
    if "support_centroid_model" in sa and "support_centroid_model" in sb:
        compared += 1
        left, right = sa["support_centroid_model"], sb["support_centroid_model"]
        if (left is None) != (right is None):
            issue("support_centroid_model", frame)
        elif left is not None:
            numeric("support_centroid_pixels", float(np.linalg.norm(np.subtract(left, right))), frame)
    if "landmarks_model" in sa and "landmarks_model" in sb:
        compared += 1
        left_points, right_points = sa["landmarks_model"], sb["landmarks_model"]
        for name in sorted(set(left_points) | set(right_points)):
            left, right = left_points.get(name), right_points.get(name)
            if (left is None) != (right is None):
                issue("landmarks_model", frame)
            elif left is not None:
                numeric("landmark_probability", abs(left[2] - right[2]), frame)
                # Where the model can't see a landmark its map is flat, and the maximum's position is noise.
                if min(left[2], right[2]) >= tolerance["landmark_visible_p"]:
                    numeric("landmark_pixels", float(np.hypot(left[0] - right[0], left[1] - right[1])), frame)
    return compared


def _compare(reference, candidate, tolerances=None):
    tolerance = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    issues, count, issue_count, new_fields = [], 0, 0, 0
    maxima = {k: 0.0 for k in ("bbox_pixels", "polyline_crop_pixels", "identity_l2", "confidence", "source_mapping_pixels", "length_px",
                               "body_origin_pixels", "support_centroid_pixels", "support_area_fraction", "time_s",
                               "landmark_probability", "landmark_pixels")}
    minimum_mask_iou = 1.0

    def issue(kind, frame):
        nonlocal issue_count
        issue_count += 1
        if len(issues) < 100:
            issues.append({"kind": kind, "frame": frame})

    def numeric(kind, value, frame):
        if not np.isfinite(value):
            issue("nonfinite_" + kind, frame)
            return
        maxima[kind] = max(maxima[kind], float(value))
        if value > tolerance[kind]:
            issue(kind, frame)

    with Path(reference).open() as left, Path(candidate).open() as right:
        for l, r in zip_longest(left, right):
            if l is None or r is None:
                issue("source_count", count)
                continue
            a, b = json.loads(l), json.loads(r)
            frame = a["frame"]
            count += 1
            if (a["frame"], a["image_path"], a["source_shape"]) != (b["frame"], b["image_path"], b["source_shape"]):
                issue("source_mapping", frame)
                continue
            if "time_s" in a and "time_s" in b:
                new_fields += 1
                if (a["time_s"] is None) != (b["time_s"] is None):
                    issue("time_s", frame)
                elif a["time_s"] is not None:
                    numeric("time_s", abs(a["time_s"] - b["time_s"]), frame)
            if len(a["stars"]) != len(b["stars"]):
                issue("star_count", frame)
                continue
            if not a["stars"]:
                continue
            boxes_a = np.array([s["crop"]["bbox_xyxy"] for s in a["stars"]])
            boxes_b = np.array([s["crop"]["bbox_xyxy"] for s in b["stars"]])
            left_ids, right_ids = linear_sum_assignment(np.abs(boxes_a[:, None] - boxes_b[None]).max(axis=2))
            for ia, ib in zip(left_ids, right_ids):
                sa, sb = a["stars"][ia], b["stars"][ib]
                numeric("bbox_pixels", np.abs(boxes_a[ia] - boxes_b[ib]).max(), frame)
                numeric("confidence", abs(sa["crop"]["confidence"] - sb["crop"]["confidence"]), frame)
                for field in ("left", "top", "side"):
                    numeric("bbox_pixels", abs(sa["crop"][field] - sb["crop"][field]), frame)
                if sa.get("star_track_id") != sb.get("star_track_id"):
                    issue("star_track_id", frame)
                new_fields += compare_body(sa, sb, frame, tolerance, issue, numeric)
                iou = mask_iou(sa["mask_polygon_source"], sb["mask_polygon_source"], a["source_shape"])
                minimum_mask_iou = min(minimum_mask_iou, iou)
                if iou < tolerance["mask_iou"]:
                    issue("mask_iou", frame)
                aa, ab = sa["arms"], sb["arms"]
                if len(aa) != len(ab):
                    issue("arm_count", frame)
                    continue
                if not aa:
                    continue
                endpoints_a = np.array([np.asarray(arm["polyline_crop"])[[0, -1]].ravel() for arm in aa])
                endpoints_b = np.array([np.asarray(arm["polyline_crop"])[[0, -1]].ravel() for arm in ab])
                arm_a, arm_b = linear_sum_assignment(np.linalg.norm(endpoints_a[:, None] - endpoints_b[None], axis=2))
                for i, j in zip(arm_a, arm_b):
                    x, y = aa[i], ab[j]
                    if x.get("track_id") != y.get("track_id"):
                        issue("arm_track_id", frame)
                    for field in ("steps", "stop_reason"):
                        if x.get(field) != y.get(field):
                            issue(field, frame)
                    px, py = np.asarray(x["polyline_crop"]), np.asarray(y["polyline_crop"])
                    if px.shape != py.shape:
                        issue("polyline_point_count", frame)
                    else:
                        numeric("polyline_crop_pixels", np.linalg.norm(px - py, axis=1).max(), frame)
                    for arm, star in ((x, sa), (y, sb)):
                        if "polyline_source" not in arm:
                            issue("missing_polyline_source", frame)
                        else:
                            points = np.asarray(arm["polyline_crop"])
                            source = np.asarray(arm["polyline_source"])
                            expected = points * star["crop"]["side"] / 512 + [star["crop"]["left"], star["crop"]["top"]]
                            if source.shape != expected.shape:
                                issue("source_shape", frame)
                            else:
                                numeric("source_mapping_pixels", np.linalg.norm(source - expected, axis=1).max(), frame)
                    ix, iy = np.asarray(x["identity"]), np.asarray(y["identity"])
                    if ix.shape != iy.shape:
                        issue("identity_shape", frame)
                    else:
                        numeric("identity_l2", np.linalg.norm(ix - iy), frame)
                    for field in ("tip_score", "base_score", "state_confidence", "length_px", "steps", "stop_reason"):
                        if field not in x or field not in y:
                            issue("missing_" + field, frame)
                        elif field in ("tip_score", "base_score", "state_confidence"):
                            numeric("confidence", abs(x[field] - y[field]), frame)
                        elif field == "length_px":
                            numeric("length_px", abs(x[field] - y[field]), frame)
    return {"pass": issue_count == 0 and count > 0, "frames": count, "issue_count": issue_count,
            "issues_first_100": issues, "maxima": maxima, "minimum_mask_iou": minimum_mask_iou,
            "new_fields_compared": new_fields,
            "tolerances": tolerance, "scope": "runtime output parity screening, not task accuracy"}


def compare(reference, candidate, tolerances=None):
    try:
        return _compare(reference, candidate, tolerances)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {"pass": False, "frames": 0, "issue_count": 1,
                "issues_first_100": [{"kind": "malformed_output", "detail": str(exc)}],
                "tolerances": {**DEFAULT_TOLERANCES, **(tolerances or {})},
                "scope": "runtime output parity screening, not task accuracy"}
