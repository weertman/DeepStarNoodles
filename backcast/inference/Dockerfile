# Version-pinned NVIDIA Blackwell-capable PyTorch/CUDA runtime.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false \
    DEEPSTARNOODLES_MODELS=/models

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 openssh-server rsync \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_*
WORKDIR /opt/deepstarnoodles-inference
COPY requirements-runtime.txt constraints-cuda128.txt ./
RUN python -m pip install --constraint constraints-cuda128.txt -r requirements-runtime.txt \
    && python -m pip install setuptools==79.0.1 wheel==0.46.3
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY licenses/ licenses/
COPY src/ src/
COPY tools/entrypoint.sh /usr/local/bin/deepstarnoodles-entrypoint
RUN python -m pip install --no-deps --no-build-isolation . \
    && python -c "import torch, torchvision; assert torch.__version__.split('+')[0] == '2.8.0'; assert torch.version.cuda == '12.8'; assert torchvision.__version__.split('+')[0] == '0.23.0'" \
    && python -m deepstarnoodles_inference smoke --device cpu \
    && rm -rf src build *.egg-info \
    && chmod 0755 /usr/local/bin/deepstarnoodles-entrypoint
WORKDIR /work
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/deepstarnoodles-entrypoint"]
CMD ["--help"]
