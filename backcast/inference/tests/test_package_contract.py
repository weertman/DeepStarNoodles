from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from deepstarnoodles_inference import artifacts, benchmark, cli
from deepstarnoodles_inference.hydra import _load_state_dict
from deepstarnoodles_inference.paths import DEFAULT_BOTSORT_CONFIG


def test_relative_manifest_and_input_paths_resolve_without_checkout(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    models = {}
    for key in ("starseg", "hydra", "tracer"):
        file = model_dir / f"{key}.pt"
        file.write_bytes(key.encode())
        models[key] = {"path": file.name, "sha256": benchmark.sha256(file)}
    models["hydra"]["architecture"] = "dinov3_convnext_base"
    manifest = model_dir / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "path_base": "manifest_directory", "models": models}))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    index = inputs / "images.csv"
    index.write_text("image_path\nimages/a.jpg\n")
    monkeypatch.chdir(tmp_path.parent)

    _, paths = benchmark.verify_models(model_dir)
    assert paths == {key: model_dir / f"{key}.pt" for key in models}
    assert list(benchmark.input_paths(index, 1)) == [inputs / "images" / "a.jpg"]
    assert artifacts.main(["verify-models", "--models", str(model_dir)]) == 0
    (model_dir / "tracer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tracer checkpoint hash"):
        benchmark.verify_models(manifest)


@pytest.mark.parametrize("command", ["archive", "live-replay"])
def test_inference_aliases_process_full_csv_without_implicit_warmup(tmp_path, monkeypatch, command):
    index = tmp_path / "images.csv"
    index.write_text("image_path\na.jpg\nb.jpg\nc.jpg\n")
    received = []
    monkeypatch.setattr(benchmark, "main", lambda argv: received.append(benchmark.parse_args(argv)) or 0)
    assert cli.main([command, "--index", str(index), "--models", "models", "--output", "out", "--device", "cpu"]) == 0
    assert received[0].mode == command
    assert received[0].images == 3
    assert received[0].warmup == 0


def test_fetch_never_starts_without_explicit_urls(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": {"starseg": {"sha256": "0" * 64}}}))
    destination = tmp_path / "downloaded"
    with pytest.raises(ValueError, match="explicit HTTPS"):
        artifacts.main(["fetch-models", "--models", str(manifest), "--destination", str(destination)])
    assert not destination.exists()


def test_checkpoint_selection_does_not_replace_exported_model_with_model_raw():
    exported, raw, ema = {"weight": torch.tensor(1)}, {"weight": torch.tensor(2)}, {"weight": torch.tensor(3)}
    assert _load_state_dict({"model": exported, "model_raw": raw}, True) is exported
    assert _load_state_dict({"model": exported, "model_raw": raw, "model_ema": ema}, True) is ema
    assert _load_state_dict({"model": exported, "model_ema": ema}, False) is exported
    assert DEFAULT_BOTSORT_CONFIG.is_file()


def test_container_launcher_keeps_cli_default_and_requires_public_key(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "tools" / "entrypoint.sh"
    env = dict(os.environ)
    env.pop("PUBLIC_KEY", None)
    result = subprocess.run(["sh", str(launcher), "runpod-worker"], env=env, capture_output=True, text=True)
    assert result.returncode == 64
    assert "requires PUBLIC_KEY" in result.stderr
    executable = tmp_path / "deepstarnoodles-inference"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    result = subprocess.run(["sh", str(launcher), "archive", "--help"], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["archive", "--help"]


def test_staged_cli_selects_archive_and_no_warmup(monkeypatch):
    from deepstarnoodles_inference import staged

    received = []
    monkeypatch.setattr(staged, "main", lambda argv: received.append(staged.parse_args(argv)) or 0)
    assert cli.main(["staged-archive", "--models", "models", "--index", "images.csv", "--output", "out", "--device", "cpu"]) == 0
    assert received[0].mode == "archive" and received[0].warmup == 0


def test_version_matches_pyproject_and_representations_is_routed(capsys):
    import re
    import deepstarnoodles_inference

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == deepstarnoodles_inference.__version__
    with pytest.raises(SystemExit) as stopped:
        cli.main(["representations", "--help"])
    assert stopped.value.code == 0 and "--untracked" in capsys.readouterr().out
