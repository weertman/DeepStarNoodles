# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""noodle_polylines: every polyline representation of the DeepStarNoodles paper figures.

Torch-free (numpy + scipy). Input is the GRU Noodle Tracer's per-arm base->tip polylines in a
pickle-free CSR container (:class:`PolylineContainer`); outputs are arrays aligned to arms or frames.

Modules: ``container`` (data model, npz I/O), ``adapters`` (released flipping / Sleepy / inference
formats), ``geometry`` (T0/T1 per arm), ``frames`` (T2 per frame), ``kinematics`` (T3 tracked),
``tracking`` (T4 FusedTracker association + Fig 6D diagnostics), ``scale`` (R0 policies), ``presets``
(per-panel parameter sets, :func:`compute_all`), ``pipeline`` (streaming). ``_reference`` holds the
slow, faithful ports of the figure scripts that the parity checks compare against.
"""
from __future__ import annotations

from . import adapters, frames, geometry, kinematics, pipeline, presets, scale, tracking
from .container import ContainerError, PolylineContainer, concat, from_arrays, from_polylines, split_at_gaps
from .pipeline import compute_many, iter_sequences
from .presets import (FIG4B, FIG4D, FIG4E, FIG5, FIG5_EXAMPLE, FIG6_FLIPPING, FIG6C, FIG6_SLEEPY, PRESETS, Preset,
                      compute_all)
from .tracking import RELEASED, TrackerConfig

__version__ = '0.1.0'
__all__ = [
    'PolylineContainer', 'ContainerError', 'concat', 'from_arrays', 'from_polylines', 'split_at_gaps',
    'adapters', 'geometry', 'frames', 'kinematics', 'tracking', 'scale', 'presets', 'pipeline',
    'Preset', 'PRESETS', 'FIG4B', 'FIG4D', 'FIG4E', 'FIG5', 'FIG5_EXAMPLE', 'FIG6_FLIPPING', 'FIG6C', 'FIG6_SLEEPY',
    'compute_all', 'compute_many', 'iter_sequences', 'TrackerConfig', 'RELEASED',
]
