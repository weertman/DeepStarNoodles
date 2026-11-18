# Copyright (c) 2026 Weertman
# SPDX-License-Identifier: MIT
"""Explicit checkpoint verification and optional manifest-directed downloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-models", "fetch-models"))
    parser.add_argument("--models", type=Path, required=True, help="Manifest JSON or directory containing manifest.json")
    parser.add_argument("--destination", type=Path, help="New directory for explicitly requested downloads")
    args = parser.parse_args(argv)
    from .benchmark import verify_models, sha256

    manifest_path = args.models / "manifest.json" if args.models.is_dir() else args.models
    if args.command == "verify-models":
        manifest, paths = verify_models(manifest_path)
        print(json.dumps({"status": "verified", "models": {key: str(path) for key, path in paths.items()},
                          "manifest_id": manifest.get("id")}))
        return 0
    if args.destination is None:
        parser.error("fetch-models requires --destination")
    manifest = json.loads(manifest_path.read_text())
    for name in ("starseg", "hydra", "tracer"):
        record = manifest["models"][name]
        if urlparse(record.get("url", "")).scheme != "https":
            raise ValueError(f"{name} requires an explicit HTTPS url in the manifest")
        if len(record.get("sha256", "")) != 64:
            raise ValueError(f"{name} requires a SHA-256 hash")
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("starseg", "hydra", "tracer"):
            record = manifest["models"][name]
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".partial", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with urlopen(record["url"], timeout=60) as response:
                    shutil.copyfileobj(response, temporary, length=8 * 1024 * 1024)
            if sha256(temporary_path) != record["sha256"]:
                raise ValueError(f"{name} download hash does not match the manifest")
            target = destination / f"{name}.pt"
            os.replace(temporary_path, target)
            record["path"] = target.name
            record.pop("url", None)
        manifest["path_base"] = "manifest_directory"
        output = destination / "manifest.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        verify_models(output)
    except Exception:
        # Only this command's newly created directory is removed on failure.
        shutil.rmtree(destination)
        raise
    print(json.dumps({"status": "downloaded_and_verified", "manifest": str(output)}))
    return 0
