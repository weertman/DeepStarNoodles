#!/bin/bash
# Install the runtime on a RunPod PyTorch pod, then fetch and verify the models. Safe to rerun.
# Usage: install.sh ROOT MODELS_MANIFEST_URL  (the code is already in ROOT/code)
set -euo pipefail
ROOT=$1
MODELS_URL=$2
cd "$ROOT"
pip_install() { python3 -m pip install --quiet --disable-pip-version-check --no-cache-dir --root-user-action=ignore "$@"; }
# The image ships a nightly Torch; the pipeline is pinned to this release.
pip_install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip_install -r code/inference/requirements-runtime.txt -c code/inference/constraints-cuda128.txt
pip_install --no-deps --force-reinstall ./code/inference
if [ ! -f models/manifest.json ]; then
  rm -rf models models.fetch
  mkdir models.fetch
  curl -fsSL --retry 3 -o models.fetch/manifest.json "$MODELS_URL"
  python3 -m deepstarnoodles_inference fetch-models --models models.fetch/manifest.json --destination models
  rm -rf models.fetch
fi
python3 -m deepstarnoodles_inference verify-models --models "$ROOT/models/manifest.json"
python3 - <<'PY'
import torch
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.cuda.is_available(), "no GPU visible to PyTorch"
print("runtime ok:", torch.__version__, torch.cuda.get_device_name(0))
PY
touch INSTALLED
