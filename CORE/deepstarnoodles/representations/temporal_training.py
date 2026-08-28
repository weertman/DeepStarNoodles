# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Training primitives for the strict-bottleneck progress temporal VAE."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Literal
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .progress import (
    ProgressRasterConfig,
    ProgressWindowConfig,
    canonical_recipe_digest,
)
from .temporal_vae import (
    ProgressTemporalVAE,
    load_progress_temporal_vae_checkpoint,
    save_progress_temporal_vae_checkpoint,
)


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class ProgressTemporalTrainingConfig:
    """Optimizer, masking, KL, validation, and checkpoint schedule."""

    steps: int = 20_000
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation: int = 1
    gradient_clip_norm: float = 1.0
    mask_ratio: float = 0.35
    mask_span_frames: int = 4
    background_weight: float = 0.05
    free_bits: float = 0.02
    fixed_beta: float = 1e-3
    kl_target_nats: float | None = 16.0
    kl_controller_rate: float = 0.01
    kl_warmup_steps: int = 1_000
    validation_interval: int = 250
    validation_batches: int = 32
    checkpoint_interval: int = 1_000
    log_interval: int = 25
    amp_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    seed: int = 20260827
    optimizer: Literal["adamw", "adam"] = "adamw"
    schedule: Literal["cosine", "constant"] = "cosine"
    adam_betas: tuple[float, float] = (0.9, 0.95)

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "gradient_accumulation",
            "mask_span_frames",
            "validation_interval",
            "validation_batches",
            "checkpoint_interval",
            "log_interval",
        ):
            _positive_integer(name, getattr(self, name))
        _nonnegative_integer("warmup_steps", self.warmup_steps)
        _nonnegative_integer("kl_warmup_steps", self.kl_warmup_steps)
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must lie in [0, steps)")
        learning_rate = _finite_number(
            "learning_rate", self.learning_rate
        )
        minimum_learning_rate = _finite_number(
            "minimum_learning_rate", self.minimum_learning_rate
        )
        if (
            learning_rate <= 0
            or minimum_learning_rate < 0
            or minimum_learning_rate > learning_rate
        ):
            raise ValueError("invalid learning-rate range")
        if _finite_number("weight_decay", self.weight_decay) < 0:
            raise ValueError("weight_decay cannot be negative")
        if _finite_number(
            "gradient_clip_norm", self.gradient_clip_norm
        ) <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        mask_ratio = _finite_number("mask_ratio", self.mask_ratio)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in (0, 1)")
        background = _finite_number(
            "background_weight", self.background_weight
        )
        if not 0.0 <= background <= 1.0:
            raise ValueError("background_weight must lie in [0, 1]")
        if _finite_number("free_bits", self.free_bits) < 0:
            raise ValueError("free_bits cannot be negative")
        if _finite_number("fixed_beta", self.fixed_beta) < 0:
            raise ValueError("fixed_beta cannot be negative")
        if (
            self.kl_target_nats is not None
            and _finite_number("kl_target_nats", self.kl_target_nats) <= 0
        ):
            raise ValueError("kl_target_nats must be positive when provided")
        if _finite_number(
            "kl_controller_rate", self.kl_controller_rate
        ) <= 0:
            raise ValueError("kl_controller_rate must be positive")
        if self.amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported AMP dtype")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.optimizer not in {"adamw", "adam"}:
            raise ValueError("optimizer must be adamw or adam")
        if self.schedule not in {"cosine", "constant"}:
            raise ValueError("schedule must be cosine or constant")
        if (
            not isinstance(self.adam_betas, (tuple, list))
            or len(self.adam_betas) != 2
        ):
            raise ValueError("adam_betas must contain two values")
        beta1 = _finite_number("adam_betas[0]", self.adam_betas[0])
        beta2 = _finite_number("adam_betas[1]", self.adam_betas[1])
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("adam_betas values must lie in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "steps",
            "warmup_steps",
            "gradient_accumulation",
            "mask_span_frames",
            "kl_warmup_steps",
            "validation_interval",
            "validation_batches",
            "checkpoint_interval",
            "log_interval",
            "seed",
        ):
            payload[name] = int(payload[name])
        for name in (
            "learning_rate",
            "minimum_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "mask_ratio",
            "background_weight",
            "free_bits",
            "fixed_beta",
            "kl_controller_rate",
        ):
            payload[name] = float(payload[name])
        payload["kl_target_nats"] = (
            None
            if self.kl_target_nats is None
            else float(self.kl_target_nats)
        )
        payload["adam_betas"] = tuple(float(value) for value in self.adam_betas)
        return payload

    def recipe_digest(self) -> str:
        return canonical_recipe_digest(
            {
                "recipe": "progress_temporal_vae_training_v1",
                "config": self.as_dict(),
            }
        )

    def digest(self) -> str:
        return self.recipe_digest()


ProgressTemporalVAETrainingConfig = ProgressTemporalTrainingConfig


class KLRateController:
    """Drive beta in log space toward a target latent code rate."""

    def __init__(
        self,
        target_nats: float,
        *,
        learning_rate: float = 0.01,
        initial_beta: float = 1e-3,
        minimum_beta: float = 1e-8,
        maximum_beta: float = 1e6,
        warmup_steps: int = 500,
    ) -> None:
        target_nats = _finite_number("target_nats", target_nats)
        learning_rate = _finite_number("learning_rate", learning_rate)
        initial_beta = _finite_number("initial_beta", initial_beta)
        minimum_beta = _finite_number("minimum_beta", minimum_beta)
        maximum_beta = _finite_number("maximum_beta", maximum_beta)
        _nonnegative_integer("warmup_steps", warmup_steps)
        if target_nats <= 0:
            raise ValueError("target_nats must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < minimum_beta <= maximum_beta:
            raise ValueError("beta bounds must satisfy 0 < minimum <= maximum")
        if initial_beta <= 0:
            raise ValueError("initial_beta must be positive")
        self.target_nats = target_nats
        self.learning_rate = learning_rate
        self.minimum_beta = minimum_beta
        self.maximum_beta = maximum_beta
        self.warmup_steps = warmup_steps
        self._log_beta = math.log(
            min(max(initial_beta, minimum_beta), maximum_beta)
        )

    @property
    def beta(self) -> float:
        return min(
            max(math.exp(self._log_beta), self.minimum_beta),
            self.maximum_beta,
        )

    def beta_at(self, step: int) -> float:
        return 0.0 if step < self.warmup_steps else self.beta

    def update(self, rate_nats: float, step: int) -> float:
        rate_nats = _finite_number("rate_nats", rate_nats)
        if step < self.warmup_steps:
            return 0.0
        error = rate_nats / self.target_nats - 1.0
        self._log_beta += self.learning_rate * min(
            max(error, -1.0), 1.0
        )
        self._log_beta = min(
            max(self._log_beta, math.log(self.minimum_beta)),
            math.log(self.maximum_beta),
        )
        return self.beta

    def state(self) -> dict[str, float]:
        return {
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "target_nats": self.target_nats,
            "warmup_steps": float(self.warmup_steps),
        }

    def state_dict(self) -> dict[str, float]:
        return {
            **self.state(),
            "minimum_beta": self.minimum_beta,
            "maximum_beta": self.maximum_beta,
        }

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        expected = {
            "target_nats": self.target_nats,
            "learning_rate": self.learning_rate,
            "warmup_steps": float(self.warmup_steps),
            "minimum_beta": self.minimum_beta,
            "maximum_beta": self.maximum_beta,
        }
        for name, value in expected.items():
            if not math.isclose(
                float(state[name]),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"KL controller {name} mismatch")
        beta = float(state["beta"])
        if (
            not math.isfinite(beta)
            or not self.minimum_beta <= beta <= self.maximum_beta
        ):
            raise ValueError("invalid KL controller beta")
        self._log_beta = math.log(beta)


def sample_temporal_reconstruction_mask(
    frame_valid: torch.Tensor,
    *,
    ratio: float,
    span_frames: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample contiguous valid spans while retaining at least one visible frame."""

    if frame_valid.ndim != 2:
        raise ValueError("frame_valid must have shape [batch, frames]")
    if frame_valid.dtype != torch.bool:
        raise TypeError("frame_valid must be boolean")
    ratio = _finite_number("ratio", ratio)
    _positive_integer("span_frames", span_frames)
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must lie in (0, 1)")
    valid_cpu = frame_valid.detach().to(device="cpu")
    mask = torch.zeros_like(valid_cpu)
    for batch_index, valid in enumerate(valid_cpu):
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        if len(positions) < 2:
            raise ValueError(
                "masked training needs at least two valid frames per context"
            )
        target = min(
            len(positions) - 1,
            max(1, round(len(positions) * ratio)),
        )
        anchors = positions[
            torch.randperm(len(positions), generator=generator)
        ]
        ordered: list[int] = []
        selected: set[int] = set()
        for anchor in anchors.tolist():
            for position in range(
                anchor, min(anchor + span_frames, len(valid))
            ):
                if valid[position] and position not in selected:
                    selected.add(position)
                    ordered.append(position)
                    if len(ordered) == target:
                        break
            if len(ordered) == target:
                break
        mask[batch_index, ordered] = True
    return mask.to(frame_valid.device)


def masked_progress_vae_loss(
    output: Mapping[str, torch.Tensor],
    progress: torch.Tensor,
    support: torch.Tensor,
    *,
    beta: float,
    free_bits: float,
    background_weight: float,
) -> dict[str, torch.Tensor]:
    """Support-weighted smooth-L1 reconstruction plus free-bits KL."""

    beta = _finite_number("beta", beta)
    free_bits = _finite_number("free_bits", free_bits)
    background_weight = _finite_number(
        "background_weight", background_weight
    )
    if beta < 0 or free_bits < 0:
        raise ValueError("beta and free_bits cannot be negative")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must lie in [0, 1]")
    if progress.shape != support.shape:
        raise ValueError("progress and support shapes differ")
    mask = output["reconstruction_mask"].bool()
    if tuple(mask.shape) != tuple(progress.shape[:2]):
        raise ValueError("reconstruction mask and targets differ")
    target = progress[mask]
    target_support = support[mask]
    logits = output["progress_logits"]
    if logits.shape != target.shape:
        raise ValueError(
            "progress prediction and selected target shapes differ"
        )
    if not len(target):
        raise ValueError("reconstruction mask selects no frame")
    prediction = torch.sigmoid(logits)
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = target_support + background_weight * (1.0 - target_support)
    reconstruction = (error * weight).sum() / weight.sum().clamp_min(1.0)
    foreground = (
        (error * target_support).sum()
        / target_support.sum().clamp_min(1.0)
    )
    inverse_support = 1.0 - target_support
    background = (
        (error * inverse_support).sum()
        / inverse_support.sum().clamp_min(1.0)
    )

    mean = output["mean"]
    logvar = output["logvar"]
    if mean.shape != logvar.shape or mean.ndim != 2:
        raise ValueError("mean and logvar must be matching matrices")
    kl_dimension = 0.5 * (
        mean.square() + logvar.exp() - 1.0 - logvar
    ).mean(dim=0)
    kl_total = kl_dimension.sum()
    kl_free = kl_dimension.clamp_min(free_bits).sum()
    total = reconstruction + beta * kl_free
    active = (kl_dimension > free_bits).sum().to(total.dtype)
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "foreground": foreground,
        "background": background,
        "kl_total": kl_total,
        "kl_free": kl_free,
        "active_dimensions": active,
        "beta": total.new_tensor(beta),
        "masked_frames": mask.sum().to(total.dtype),
    }


class _CyclingLoader:
    def __init__(
        self, loader: Iterable[Mapping[str, torch.Tensor]]
    ) -> None:
        self.loader = loader
        self.iterator = iter(loader)
        self.epoch = 0

    def next(self) -> Mapping[str, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            sampler = getattr(self.loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise ValueError("training loader is empty") from exc


def _move_progress_batch(
    raw_batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    model: ProgressTemporalVAE,
) -> dict[str, torch.Tensor]:
    missing = {"progress", "support", "frame_valid"} - set(raw_batch)
    if missing:
        raise ValueError(f"progress batch is missing {sorted(missing)}")
    progress = torch.as_tensor(raw_batch["progress"])
    support = torch.as_tensor(raw_batch["support"])
    frame_valid = torch.as_tensor(raw_batch["frame_valid"])
    if progress.ndim == 4:
        progress = progress.unsqueeze(0)
        support = support.unsqueeze(0)
        frame_valid = frame_valid.unsqueeze(0)
    expected = (
        progress.shape[0],
        model.config.context_frames,
        1,
        model.config.resolution,
        model.config.resolution,
    )
    if tuple(progress.shape) != expected:
        raise ValueError(f"progress must have shape {expected}")
    if support.shape != progress.shape:
        raise ValueError("support must match progress")
    if tuple(frame_valid.shape) != expected[:2]:
        raise ValueError("frame_valid must match progress batch and time")
    frame_valid = frame_valid.to(
        device=device, dtype=torch.bool, non_blocking=True
    )
    moved: dict[str, torch.Tensor] = {}
    for name, source in (("progress", progress), ("support", support)):
        value = source.to(device, non_blocking=True)
        if value.dtype == torch.uint8:
            value = value.to(torch.float32).mul_(1.0 / 255.0)
        elif value.is_floating_point():
            value = value.to(torch.float32)
        else:
            raise TypeError(f"{name} must be floating point or uint8")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        value = value.masked_fill(
            ~frame_valid[..., None, None, None], 0.0
        )
        moved[name] = value
    moved["frame_valid"] = frame_valid
    return moved


def _learning_rate_factor(
    step: int, config: ProgressTemporalTrainingConfig
) -> float:
    if step < config.warmup_steps:
        return (step + 1) / max(config.warmup_steps, 1)
    if config.schedule == "constant":
        return 1.0
    progress = (step - config.warmup_steps) / max(
        config.steps - config.warmup_steps - 1, 1
    )
    progress = min(max(progress, 0.0), 1.0)
    minimum = config.minimum_learning_rate / config.learning_rate
    return minimum + (1.0 - minimum) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _make_optimizer(
    model: ProgressTemporalVAE,
    config: ProgressTemporalTrainingConfig,
) -> torch.optim.Optimizer:
    arguments = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": tuple(config.adam_betas),
    }
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **arguments)
    return torch.optim.Adam(model.parameters(), **arguments)


def _make_controller(
    config: ProgressTemporalTrainingConfig,
) -> KLRateController | None:
    if config.kl_target_nats is None:
        return None
    return KLRateController(
        config.kl_target_nats,
        learning_rate=config.kl_controller_rate,
        initial_beta=max(config.fixed_beta, 1e-8),
        warmup_steps=config.kl_warmup_steps,
    )


def _beta(
    step: int,
    config: ProgressTemporalTrainingConfig,
    controller: KLRateController | None,
) -> float:
    if controller is not None:
        return controller.beta_at(step)
    if step >= config.kl_warmup_steps:
        return config.fixed_beta
    return config.fixed_beta * step / max(config.kl_warmup_steps, 1)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _training_state(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    controller: KLRateController | None,
    mask_generator: torch.Generator,
    step: int,
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "kl_controller": (
            None if controller is None else controller.state_dict()
        ),
        "mask_generator_state": mask_generator.get_state(),
        "step": int(step),
        "best_validation_loss": float(best_validation_loss),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def _normalize_saved_training_config(
    saved: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = ProgressTemporalTrainingConfig().as_dict()
    defaults.update(dict(saved))
    if isinstance(defaults.get("adam_betas"), list):
        defaults["adam_betas"] = tuple(defaults["adam_betas"])
    return ProgressTemporalTrainingConfig(**defaults).as_dict()


def _restore_training(
    path: str | Path,
    *,
    model: ProgressTemporalVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    controller: KLRateController | None,
    mask_generator: torch.Generator,
    config: ProgressTemporalTrainingConfig,
) -> tuple[int, float]:
    loaded = load_progress_temporal_vae_checkpoint(
        path, "cpu", return_info=True
    )
    if loaded.model.config != model.config:
        raise ValueError("checkpoint model configuration mismatch")
    if loaded.training_config is None or loaded.training_state is None:
        raise ValueError("resume checkpoint has no training state")
    normalized = _normalize_saved_training_config(loaded.training_config)
    if canonical_recipe_digest(normalized) != canonical_recipe_digest(
        config.as_dict()
    ):
        raise ValueError("checkpoint training configuration mismatch")
    state = loaded.training_state
    model.load_state_dict(loaded.model.state_dict(), strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    controller_state = state.get("kl_controller")
    if (controller is None) != (controller_state is None):
        raise ValueError("checkpoint KL controller mismatch")
    if controller is not None:
        controller.load_state_dict(controller_state)
    mask_generator.set_state(state["mask_generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_state = state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    return int(state["step"]), float(state["best_validation_loss"])


@torch.no_grad()
def validate_progress_temporal_vae(
    model: ProgressTemporalVAE,
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device,
    config: ProgressTemporalTrainingConfig,
    beta: float,
) -> dict[str, float]:
    """Evaluate deterministic posterior means and deterministic masks."""

    selected_device = torch.device(device)
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1_000_000)
    totals: dict[str, torch.Tensor] = {}
    batches = 0
    use_amp = (
        selected_device.type == "cuda"
        and config.amp_dtype != "float32"
    )
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.amp_dtype]
    for raw_batch in loader:
        batch = _move_progress_batch(
            raw_batch, device=selected_device, model=model
        )
        mask = sample_temporal_reconstruction_mask(
            batch["frame_valid"],
            ratio=config.mask_ratio,
            span_frames=config.mask_span_frames,
            generator=generator,
        )
        with torch.autocast(
            device_type=selected_device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            output = model(
                batch["progress"],
                batch["frame_valid"],
                mask,
                sample=False,
            )
            losses = masked_progress_vae_loss(
                output,
                batch["progress"],
                batch["support"],
                beta=beta,
                free_bits=config.free_bits,
                background_weight=config.background_weight,
            )
        for name, value in losses.items():
            totals[name] = (
                totals.get(name, torch.zeros_like(value))
                + value.detach()
            )
        batches += 1
        if batches == config.validation_batches:
            break
    if not batches:
        raise ValueError("validation loader is empty")
    return {
        name: float((value / batches).float().cpu())
        for name, value in totals.items()
    }


def train_progress_temporal_vae(
    model: ProgressTemporalVAE,
    train_loader: Iterable[Mapping[str, torch.Tensor]],
    validation_loader: Iterable[Mapping[str, torch.Tensor]],
    output_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    config: ProgressTemporalTrainingConfig = ProgressTemporalTrainingConfig(),
    resume: str | Path | None = None,
    preprocess_config: ProgressRasterConfig | None = None,
    window_config: ProgressWindowConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, validate, atomically checkpoint, and optionally resume a VAE."""

    if not isinstance(model, ProgressTemporalVAE):
        raise TypeError("model must be a ProgressTemporalVAE")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = model.to(selected_device)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_factor(step, config)
    )
    use_amp = (
        selected_device.type == "cuda"
        and config.amp_dtype != "float32"
    )
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.amp_dtype]
    scaler = torch.amp.GradScaler(
        selected_device.type,
        enabled=use_amp and amp_dtype == torch.float16,
    )
    controller = _make_controller(config)
    mask_generator = torch.Generator().manual_seed(config.seed)
    cycling = _CyclingLoader(train_loader)
    start_step = 0
    best_validation_loss = float("inf")
    if resume is not None:
        start_step, best_validation_loss = _restore_training(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            controller=controller,
            mask_generator=mask_generator,
            config=config,
        )
        if start_step > config.steps:
            raise ValueError("resume checkpoint exceeds configured steps")
    history_path = output / "training_history.json"
    history: list[dict[str, Any]] = []
    if start_step and history_path.is_file():
        loaded_history = json.loads(
            history_path.read_text(encoding="utf-8")
        )
        if not isinstance(loaded_history, list):
            raise ValueError("training history is malformed")
        history = loaded_history
    clean_metadata = dict(metadata or {})

    for step in range(start_step + 1, config.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        beta = _beta(step, config, controller)
        for _ in range(config.gradient_accumulation):
            batch = _move_progress_batch(
                cycling.next(), device=selected_device, model=model
            )
            mask = sample_temporal_reconstruction_mask(
                batch["frame_valid"],
                ratio=config.mask_ratio,
                span_frames=config.mask_span_frames,
                generator=mask_generator,
            )
            with torch.autocast(
                device_type=selected_device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                predicted = model(
                    batch["progress"],
                    batch["frame_valid"],
                    mask,
                    sample=True,
                )
                losses = masked_progress_vae_loss(
                    predicted,
                    batch["progress"],
                    batch["support"],
                    beta=beta,
                    free_bits=config.free_bits,
                    background_weight=config.background_weight,
                )
                scaled_loss = (
                    losses["loss"] / config.gradient_accumulation
                )
            scaler.scale(scaled_loss).backward()
            for name, value in losses.items():
                accumulated[name] = (
                    accumulated.get(name, torch.zeros_like(value))
                    + value.detach()
                )
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        averaged = {
            name: value / config.gradient_accumulation
            for name, value in accumulated.items()
        }
        if controller is not None:
            controller.update(
                float(averaged["kl_total"].float().cpu()), step
            )
        record: dict[str, Any] = {
            "step": step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": float(gradient_norm),
            "train": {
                name: float(value.float().cpu())
                for name, value in averaged.items()
            },
        }

        should_validate = (
            step % config.validation_interval == 0
            or step == config.steps
        )
        if should_validate:
            validation = validate_progress_temporal_vae(
                model,
                validation_loader,
                device=selected_device,
                config=config,
                beta=_beta(step, config, controller),
            )
            record["validation"] = validation
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                save_progress_temporal_vae_checkpoint(
                    output / "best.pt",
                    model,
                    checkpoint_type="training",
                    metadata=clean_metadata,
                    preprocess_config=preprocess_config,
                    window_config=window_config,
                    training_config=config.as_dict(),
                    training_state=_training_state(
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        controller=controller,
                        mask_generator=mask_generator,
                        step=step,
                        best_validation_loss=best_validation_loss,
                    ),
                )
        history.append(record)
        should_checkpoint = (
            step % config.checkpoint_interval == 0
            or step == config.steps
        )
        if should_checkpoint:
            save_progress_temporal_vae_checkpoint(
                output / "latest.pt",
                model,
                checkpoint_type="training",
                metadata=clean_metadata,
                preprocess_config=preprocess_config,
                window_config=window_config,
                training_config=config.as_dict(),
                training_state=_training_state(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    controller=controller,
                    mask_generator=mask_generator,
                    step=step,
                    best_validation_loss=best_validation_loss,
                ),
            )
            _atomic_json(history_path, history)
        if step % config.log_interval == 0:
            print(
                "PROGRESS_VAE_STEP "
                f"step={step}/{config.steps} "
                f"loss={float(averaged['loss'].float().cpu()):.6f} "
                f"reconstruction="
                f"{float(averaged['reconstruction'].float().cpu()):.6f} "
                f"kl={float(averaged['kl_total'].float().cpu()):.3f} "
                f"beta={beta:.6g} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

    result = {
        "steps": config.steps,
        "start_step": start_step,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "output_directory": str(output.resolve()),
        "latest_checkpoint": str((output / "latest.pt").resolve()),
        "best_checkpoint": str((output / "best.pt").resolve()),
    }
    _atomic_json(
        output / "complete.json",
        {name: value for name, value in result.items() if name != "history"},
    )
    return result


def fine_tune_progress_temporal_vae(
    source: ProgressTemporalVAE | str | Path,
    train_loader: Iterable[Mapping[str, torch.Tensor]],
    validation_loader: Iterable[Mapping[str, torch.Tensor]],
    output_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    config: ProgressTemporalTrainingConfig = ProgressTemporalTrainingConfig(),
    resume: str | Path | None = None,
    preprocess_config: ProgressRasterConfig | None = None,
    window_config: ProgressWindowConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Warm-start all VAE parameters, then use the standard training loop."""

    if isinstance(source, ProgressTemporalVAE):
        model = source
        source_digest = model.checkpoint_sha256
        inherited_preprocess = model.checkpoint_preprocess_config
        inherited_window = model.checkpoint_window_config
    else:
        loaded = load_progress_temporal_vae_checkpoint(
            source, "cpu", return_info=True
        )
        model = loaded.model
        source_digest = loaded.checkpoint_sha256
        inherited_preprocess = loaded.preprocess_config
        inherited_window = loaded.window_config
    combined_metadata = dict(metadata or {})
    if source_digest is not None:
        combined_metadata.setdefault(
            "fine_tuned_from_checkpoint_sha256", source_digest
        )
    return train_progress_temporal_vae(
        model,
        train_loader,
        validation_loader,
        output_directory,
        device=device,
        config=config,
        resume=resume,
        preprocess_config=preprocess_config or inherited_preprocess,
        window_config=window_config or inherited_window,
        metadata=combined_metadata,
    )


__all__ = [
    "KLRateController",
    "ProgressTemporalTrainingConfig",
    "ProgressTemporalVAETrainingConfig",
    "fine_tune_progress_temporal_vae",
    "masked_progress_vae_loss",
    "sample_temporal_reconstruction_mask",
    "train_progress_temporal_vae",
    "validate_progress_temporal_vae",
]
