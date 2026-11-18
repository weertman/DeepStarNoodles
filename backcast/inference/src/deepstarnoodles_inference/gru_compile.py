# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Experimental: ``torch.compile`` the fixed-slot GRU step before CUDA-graph capture.

Inductor fuses the step's elementwise and indexing work into a few Triton
kernels; convolutions and matrix products still call cuDNN and cuBLAS. Fused
arithmetic rounds differently from eager kernels, so outputs are not
byte-identical to ``--gru-execution graph``: measure strict parity and
validation accuracy before selecting it.

Triton builds one small C helper (``cuda_utils``) on first use. Runtime images
without a C compiler or Python headers can supply prebuilt helpers through
``DEEPSTARNOODLES_TRITON_PREBUILT``: a JSON manifest mapping each module name to
the SHA-256 of its C source and of a prebuilt shared object beside the
manifest. A module whose source or binary differs from the manifest is never
substituted, and a host with both always builds its own.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sysconfig

import torch

PREBUILT_ENV = "DEEPSTARNOODLES_TRITON_PREBUILT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prebuilt_build(manifest_path: Path):
    """A Triton ``knobs.build.impl`` serving the prebuilt modules of one manifest."""
    manifest_path = Path(manifest_path)
    modules = json.loads(manifest_path.read_text(encoding="utf-8"))["modules"]

    def build(name, src, srcdir, library_dirs, include_dirs, libraries):
        entry = modules.get(name)
        if entry is None or _sha256(src) != entry["source_sha256"]:
            raise RuntimeError(f"no C compiler and no prebuilt Triton module {name!r} for this source")
        prebuilt = manifest_path.parent / entry["file"]
        if prebuilt.name != entry["file"] or _sha256(prebuilt) != entry["sha256"]:
            raise RuntimeError(f"prebuilt Triton module {name!r} differs from its manifest")
        target = Path(srcdir) / prebuilt.name
        shutil.copyfile(prebuilt, target)
        return str(target)

    return build


def install_prebuilt_build() -> bool:
    """Use the prebuilt helpers only when this host cannot build them itself."""
    manifest = os.environ.get(PREBUILT_ENV)
    compiler = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    headers = (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file()
    if not manifest or (compiler and headers):
        return False
    from triton import knobs
    knobs.build.impl = prebuilt_build(Path(manifest))
    return True


def static_kernel_configs() -> None:
    """Fix Inductor's kernel configurations instead of timing candidates at startup.

    Autotune picks by measured speed, so each process can get different kernels
    and round differently: compiled outputs then change from run to run. With
    fixed configurations, repeated runs on one GPU type give the same bytes.
    """
    torch._inductor.config.triton.autotune_pointwise = False
    torch._inductor.config.triton.autotune_cublasLt = False
    # Concurrent archive shards must not share Inductor's local tuning cache files.
    torch._inductor.config.autotune_local_cache = False


def compile_hydra(predictor) -> None:
    """Replace ``predictor.model`` with its compiled forward, built on the first call.

    Inductor fuses the normalization, activation and layer-scale arithmetic between
    convolutions, which is most of a ConvNeXt forward at this resolution. Like the
    compiled GRU step, it rounds differently from eager kernels.
    """
    install_prebuilt_build()
    static_kernel_configs()
    predictor.model = torch.compile(predictor.model)


@functools.lru_cache(maxsize=None)
def compiled_rollout_step():
    """One shared compiled ``rollout_step`` per process, compiled lazily per specialization."""
    from .noodle_tracer_graph import rollout_step
    install_prebuilt_build()
    # One specialization each for the first and later steps, plus a dynamic walker count.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    static_kernel_configs()
    return torch.compile(rollout_step, dynamic=True)
