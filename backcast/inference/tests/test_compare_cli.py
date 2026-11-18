"""Public comparison CLI preserves the fixed gate and portable evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from deepstarnoodles_inference import cli, compare_cli, validation


@pytest.fixture
def pair(tmp_path):
    arm = {"polyline_crop": [[10.0, 20.0], [30.0, 40.0]],
           "polyline_source": [[10.0, 20.0], [30.0, 40.0]],
           "identity": [0.1, 0.2], "steps": 2, "stop_reason": "base",
           "tip_score": 0.8, "base_score": 0.9, "state_confidence": 0.9,
           "length_px": 28.284}
    record = {"frame": 0, "image_path": "/private/source/frame.png",
              "source_shape": [512, 512], "stars": [{
                  "crop": {"left": 0.0, "top": 0.0, "side": 512.0,
                           "confidence": 0.9, "bbox_xyxy": [0, 0, 50, 50]},
                  "mask_polygon_source": None, "arms": [arm]}]}
    paths = [tmp_path / name for name in ("reference.jsonl", "candidate.jsonl")]
    for path in paths:
        path.write_text(json.dumps(record) + "\n")
    return paths, record


def args(paths):
    return ["compare", "--reference", str(paths[0]), "--candidate", str(paths[1])]


def test_route_pass_hashes_and_optional_output(pair, tmp_path, capsys):
    paths, _ = pair
    output = tmp_path / "parity.json"
    assert cli.main([*args(paths), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report["parity"]["pass"] is True
    assert report["parity"]["tolerances"] == validation.DEFAULT_TOLERANCES
    assert report["input_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(("reference", "candidate"), paths)}
    assert "/private/" not in output.read_text()
    assert str(tmp_path) not in output.read_text()


def test_changed_trace_fails_with_same_geometry_and_counts(pair, capsys):
    paths, record = pair
    changed = deepcopy(record)
    changed["stars"][0]["arms"][0]["steps"] += 1
    paths[1].write_text(json.dumps(changed) + "\n")
    assert cli.main(args(paths)) == 1
    parity = json.loads(capsys.readouterr().out)["parity"]
    assert parity["pass"] is False
    assert {issue["kind"] for issue in parity["issues_first_100"]} == {"steps"}


@pytest.mark.parametrize("corruption", ["{broken", '{"frame":"/private/patient/path"}', "\n"])
def test_malformed_fails_without_exception_details(pair, capsys, corruption):
    paths, _ = pair
    paths[1].write_text(corruption)
    assert cli.main(args(paths)) == 1
    text = capsys.readouterr().out
    report = json.loads(text)
    assert report["parity"]["issues_first_100"] == [{"kind": "malformed_output"}]
    assert "/private/" not in text
    assert "detail" not in text


def test_missing_input_returns_failed_report_and_available_hash(pair, capsys):
    paths, _ = pair
    paths[1].unlink()
    assert cli.main(args(paths)) == 1
    text = capsys.readouterr().out
    report = json.loads(text)
    assert report["input_sha256"]["candidate"] is None
    assert report["input_sha256"]["reference"] is not None
    assert report["parity"]["issues_first_100"] == [{"kind": "unreadable_input"}]
    assert str(paths[1]) not in text


@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_existing_output_never_overwritten(pair, tmp_path, capsys, monkeypatch, dangling_symlink):
    paths, _ = pair
    output = tmp_path / "existing.json"
    if dangling_symlink:
        output.symlink_to(tmp_path / "missing-target")
    else:
        output.write_text("KEEP")
    def forbidden_hash(path):
        pytest.fail("Existing output should fail before reading inputs")
    monkeypatch.setattr(compare_cli, "_sha256", forbidden_hash)
    assert cli.main([*args(paths), "--output", str(output)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert output.is_symlink() if dangling_symlink else output.read_text() == "KEEP"


def test_output_appearance_race_cannot_overwrite(pair, tmp_path, monkeypatch, capsys):
    paths, _ = pair
    output = tmp_path / "race.json"
    real_compare = validation.compare
    def with_race(*values):
        output.write_text("OTHER WRITER")
        return real_compare(*values)
    monkeypatch.setattr(validation, "compare", with_race)
    assert cli.main([*args(paths), "--output", str(output)]) == 1
    assert output.read_text() == "OTHER WRITER"
    assert "not overwritten" in capsys.readouterr().err


def test_changing_inputs_cannot_pass_with_stale_hashes(pair, monkeypatch, capsys):
    paths, _ = pair
    real_compare = validation.compare
    def with_change(*values):
        result = real_compare(*values)
        paths[1].write_text("changed after comparison\n")
        return result
    monkeypatch.setattr(validation, "compare", with_change)
    assert cli.main(args(paths)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["parity"]["issues_first_100"] == [{"kind": "input_changed_during_comparison"}]


def test_tolerances_cannot_be_relaxed_on_cli(pair):
    paths, _ = pair
    with pytest.raises(SystemExit) as failure:
        cli.main([*args(paths), "--confidence-tolerance", "1"])
    assert failure.value.code == 2


def test_module_entrypoint_exit_codes_and_no_model_imports(pair):
    paths, _ = pair
    process = subprocess.run(
        [sys.executable, "-c", "from deepstarnoodles_inference.cli import main; "
         "import sys; result=main(sys.argv[1:]); "
         "assert 'torch' not in sys.modules; raise SystemExit(result)", *args(paths)],
        capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["parity"]["pass"] is True
    paths[1].write_text("malformed")
    process = subprocess.run([sys.executable, "-m", "deepstarnoodles_inference", *args(paths)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 1
    assert json.loads(process.stdout)["parity"]["pass"] is False


def test_dev10_reference_against_schema3_candidate_passes(pair, capsys):
    paths, record = pair
    upgraded = deepcopy(record)
    upgraded["stars"][0].update(body_origin_model=[256.0, 256.0], body_origin_code=2,
                                support_area_model_px=0, support_centroid_model=None)
    upgraded.update(time_s=None, fps=None, schema_version=3)
    paths[1].write_text(json.dumps(upgraded) + "\n")
    assert cli.main(args(paths)) == 0
    parity = json.loads(capsys.readouterr().out)["parity"]
    assert parity["pass"] is True and parity["new_fields_compared"] == 0
    changed = deepcopy(upgraded)
    changed["stars"][0]["body_origin_code"] = 0
    paths[0].write_text(json.dumps(upgraded) + "\n")
    paths[1].write_text(json.dumps(changed) + "\n")
    assert cli.main(args(paths)) == 1
    parity = json.loads(capsys.readouterr().out)["parity"]
    assert {issue["kind"] for issue in parity["issues_first_100"]} == {"body_origin_code"}
