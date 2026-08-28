# Configurable behavior representations

DeepStarNoodles exposes three independent representation layers:

- a stateless bag of decoded arm shapes;
- causal motion summaries from continuously tracked arms;
- a learned temporal VAE over dense Hydra progress fields.

The published DeepNoodleTracker/Sleepy recipes are defaults. Every public
configuration can be replaced, and every result carries a deterministic recipe
digest so analyses can distinguish customized representations.

## Static arm bags

`arm_bag` accepts base-to-tip polylines or the arm dictionaries returned by the
Noodle Tracer:

```python
from deepstarnoodles import ArmBagConfig, arm_bag

default = arm_bag(arms)
assert default.values.shape == (102,)

custom = arm_bag(
    arms,
    ArmBagConfig(samples=32),
    weights=arm_confidences,
)
assert custom.values.shape == (134,)
```

The default resamples each arm to 24 points, aligns its base-tip chord to +x,
divides shape coordinates by that arm's arc length, appends normalized length,
tip radius, and mean absolute turning, then mean/std-pools across arms.
`samples=N` produces `4*N + 6` features. Frame-median scales are the default;
use `scale_policy="provided"` with explicit `length_scale` and `radius_scale`
for a calibrated alternative.

The result is fixed-width and permutation-invariant, but it is not invariant to
adding or removing detected arms: the pooled moments and frame medians change.

## Tracked arm motion

`arm_motion` accepts typed `TrackedArmFrame` records or mappings containing a
timestamp, an observation flag, and tracked arms. Existing tracking JSON can be
adapted directly:

```python
import json
from deepstarnoodles.representations import (
    ArmMotionConfig,
    arm_motion,
    tracked_frames_from_video_payload,
)

payload = json.load(open("tracks.json"))
frames = tracked_frames_from_video_payload(payload, fps=30.0)
motion = arm_motion(
    frames,
    ArmMotionConfig(lags_seconds=(0.5, 1, 3), distal_points=8),
)
```

The default output has shape `[frame, 3, 5]` for lags 1, 3, and 5 seconds and
features distal mean, q90, maximum, participation fraction, and top-two share.
Primary values require same-ID continuity through every intervening frame.
Parallel strict values also require endpoint identity embeddings to remain
within `identity_gate`.

Lag comparisons are causal. Default `sequence_median` length normalization is
an offline whole-sequence statistic; use `scale_policy="provided"` for a fixed
calibration suitable for streaming.

## Progress temporal VAE

The default temporal model consumes raw Hydra channels 0 and 4. The helper
constructs:

```text
hard support = sigmoid(raw support) > 0.5
support      = sigmoid(raw support) * hard support
progress     = sigmoid(raw progress) * hard support * support
```

It then recenters every frame on support, maps source coordinates through an
explicit affine, scales by an individual's characteristic support radius, and
renders a 64×64 raster.

```python
import torch
from deepstarnoodles import ProgressSequenceEncoder
from deepstarnoodles.representations import (
    ProgressCanonicalizationInput,
    estimate_characteristic_scale,
    progress_from_hydra,
)

activated = progress_from_hydra(raw_hydra_sequence)
scale = estimate_characteristic_scale(
    activated.support,
    crop_left_top_side=crop_left_top_side,
    model_size=512,
)
geometry = ProgressCanonicalizationInput.source_axes(
    crop_left_top_side,
    model_size=512,
    characteristic_scale=scale,
)

encoder = ProgressSequenceEncoder(device="cuda:0")
latents = encoder.encode_hydra(raw_hydra_sequence, timestamps, geometry)
assert latents.mean.shape[1] == 64
```

`source_axes` preserves image heading and is a convenience mode, not an
aquarium calibration. For comparable multi-camera analyses, supply a calibrated
`source_to_world_affine` and measure `characteristic_scale` in those world
units.

The released model uses one frame per second and a centered context from −30
through +29 seconds. Its output is therefore acausal. A trailing window can be
requested with `ProgressWindowConfig(centered=False)`, but it is an inference
experiment with weights trained under the centered-window recipe.

The sequence encoder calculates each frame's CNN token once and reuses it for
all overlapping windows. It returns posterior means, log-variances, validity,
window coverage, timestamps, offsets, checkpoint provenance, and a recipe
digest.

## Training and fine-tuning

Architecture and objective parameters are public configuration objects:

```python
from torch.utils.data import DataLoader
from deepstarnoodles.representations import (
    ProgressTemporalTrainingConfig,
    ProgressTemporalVAE,
    ProgressTemporalVAEConfig,
    ProgressWindowDataset,
    train_progress_temporal_vae,
)

model = ProgressTemporalVAE(
    ProgressTemporalVAEConfig(
        context_frames=90,
        latent_dim=96,
        transformer_layers=6,
    )
)
dataset = ProgressWindowDataset(
    canonical_progress,
    canonical_support,
    frame_valid,
    context_frames=90,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

train_progress_temporal_vae(
    model,
    loader,
    validation_loader,
    "RUNS/my_progress_vae",
    device="cuda:0",
    config=ProgressTemporalTrainingConfig(mask_ratio=0.25),
)
```

Changing model width, latent size, resolution, or context length changes tensor
shapes and requires new training. Device, batch sizes, output stride, and
minimum valid coverage are runtime settings. `fine_tune_progress_temporal_vae`
warm-starts from any compatible public or user checkpoint.

Training checkpoints include optimizer and exact-resume state. Use
`export_inference_checkpoint` to produce the smaller schema-versioned artifact
used by `ProgressSequenceEncoder`.

## Released default weights

The optional checkpoint is:

```text
MODELS/temporal_vae/dnt_progress_temporal_vae_60s_sleepy.pt
SHA-256 182de7d66d3b670c94e25d4f78070d7ffa5851acd76200cc7c059849b1abddc9
```

Download it with the other first-party weights:

```bash
hf download weertman/deepstarnoodles --local-dir MODELS
```

This is an inference-only export of the DeepNoodleTracker checkpoint at step
10,250. It was trained on dense fields from Hydra checkpoint
`3f691daa14a891a5f9ad163355d969f8922892c662d4d9b7864a54fd815eecee`.
The currently released Hydra uses the same channel contract but a different
weight set; the combination is available for experimentation, not yet a
scientifically validated transfer result.

## Sidecar output

Use `save_representation_sidecar` to store aligned bag, motion, and temporal
arrays in one versioned, pickle-free NPZ. Large vectors remain outside
`tracks.json`; the sidecar includes feature names, masks, recipe metadata, and
checkpoint hashes.
