# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Checkpoint contracts extracted from the research evaluator."""
from __future__ import annotations
import torch
from .model import CHANNEL_NAMES, CHANNEL_NAMES_WITHOUT_CENTERLINE

def infer_embedding_dim(checkpoint: dict | object) -> int:
    """Infer appended embedding channels from config, then embedding-head state."""
    if not isinstance(checkpoint, dict):
        return 0
    spec = checkpoint.get('output_spec')
    if not isinstance(spec, dict):
        config = checkpoint.get('config')
        spec = config.get('output_spec') if isinstance(config, dict) else None
    if isinstance(spec, dict):
        embedding = spec.get('embedding')
        if isinstance(embedding, dict) and embedding.get('dim') is not None:
            return max(0, int(embedding['dim']))
    config = checkpoint.get('config')
    if isinstance(config, dict) and config.get('embedding_dim') is not None:
        dim = int(config['embedding_dim'])
        if dim >= 0:
            return dim
    state = checkpoint.get('model', checkpoint)
    if isinstance(state, dict):
        for key in ('embedding_head.out.weight', 'module.embedding_head.out.weight'):
            weight = state.get(key)
            if isinstance(weight, torch.Tensor):
                return int(weight.shape[0])
    return 0


def output_slices(spec: dict | None, dense_channels: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve dense geometry/embedding slices from an explicit output contract."""
    if spec is None:
        if dense_channels < 10:
            raise ValueError(f'dense output has {dense_channels} channels; expected at least 10')
        return (0, 10), (10, dense_channels)
    if not isinstance(spec, dict):
        raise ValueError('output_spec must be a dictionary')

    def validated(name: str) -> tuple[int, int]:
        section = spec.get(name)
        if not isinstance(section, dict):
            raise ValueError(f'output_spec.{name} must be a dictionary')
        value = section.get('slice')
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f'output_spec.{name}.slice must contain two integers')
        start, stop = value
        if start < 0 or stop < start or stop > dense_channels:
            raise ValueError(
                f'output_spec.{name}.slice {value} is out of bounds for {dense_channels} channels'
            )
        return start, stop

    geometry_slice = validated('geometry')
    embedding_slice = validated('embedding')
    geometry = spec['geometry']
    names = geometry.get('names')
    allowed_names = {tuple(CHANNEL_NAMES), tuple(CHANNEL_NAMES_WITHOUT_CENTERLINE)}
    if names is None and geometry_slice == (0, 10):
        names = CHANNEL_NAMES
    if not isinstance(names, (list, tuple)) or tuple(names) not in allowed_names:
        raise ValueError('output_spec.geometry.names is not a supported dense channel contract')
    if geometry_slice != (0, len(names)):
        raise ValueError('output_spec.geometry.slice does not match geometry.names')
    if embedding_slice[0] < geometry_slice[1]:
        raise ValueError('output_spec embedding slice overlaps geometry')
    embedding = spec['embedding']
    if embedding.get('dim') is not None:
        dim = embedding['dim']
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError('output_spec.embedding.dim must be a non-negative integer')
        if embedding_slice[1] - embedding_slice[0] != dim:
            raise ValueError('output_spec embedding slice and dim disagree')
    return geometry_slice, embedding_slice


def checkpoint_output_spec(checkpoint: dict | object) -> dict | None:
    if not isinstance(checkpoint, dict):
        return None
    if isinstance(checkpoint.get('output_spec'), dict):
        return checkpoint['output_spec']
    config = checkpoint.get('config')
    return config.get('output_spec') if isinstance(config, dict) and isinstance(config.get('output_spec'), dict) else None


def infer_count_classes(checkpoint: dict | object) -> int:
    """Resolve optional image-level count classes without inspecting dense channels."""
    spec = checkpoint_output_spec(checkpoint)
    if isinstance(spec, dict):
        count_spec = spec.get('auxiliary', {}).get('count_logits')
        if isinstance(count_spec, dict) and count_spec.get('classes') is not None:
            return max(0, int(count_spec['classes']))
    if isinstance(checkpoint, dict):
        config = checkpoint.get('config')
        if isinstance(config, dict) and config.get('count_classes') is not None:
            return max(0, int(config['count_classes']))
        state = checkpoint.get('model', checkpoint)
        if isinstance(state, dict):
            weight = state.get('count_head.weight')
            if isinstance(weight, torch.Tensor):
                return int(weight.shape[0])
    return 0
