"""Command line: python -m backcast doctor|plan|run|status|report|stop|terminate|retry-failed --config fleet.toml"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import __version__, footage
from .config import load
from .controller import Controller, ledger_for, load_pilot, probe_all, scan, terminate_open_pods
from .ledger import ASSIGNED, DONE, FAILED, PENDING
from .pod import INFERENCE
from .runpod_api import RunPod

REQUIRED_TOOLS = ("ssh", "ssh-keygen", "rsync", "zstd", "ffprobe")
# Measured on the synthetic lab videos: raw records about 78 KB per frame, about 18 KB with zstd at 2.76 stars per frame.
OUTPUT_KB_PER_FRAME = 18


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def doctor(args, config) -> int:
    problems = 0

    def check(ok: bool, text: str, fix: str = "") -> None:
        nonlocal problems
        problems += not ok
        print(("ok    " if ok else "FAIL  ") + text + ("" if ok or not fix else f"\n      fix: {fix}"))

    for tool in REQUIRED_TOOLS:
        check(shutil.which(tool) is not None, f"{tool} on PATH", f"install {tool} (apt install openssh-client rsync zstd ffmpeg, or conda install)")
    check((INFERENCE / "pyproject.toml").exists(), "vendored inference package present", "reclone the repository")
    try:
        api = RunPod()
        account = api.account()
        check(True, f"RunPod key works; balance ${account['clientBalance']:.2f}")
        check(account["clientBalance"] >= config.max_total_usd, f"balance covers the ${config.max_total_usd:.0f} budget", "add credit or lower max_total_usd")
        print(f"info  gpu_family = {config.gpu_family}: " + ("the fleet mixes chip families; see docs/speed_test.md" if config.gpu_family == "any"
                                                               else "the fleet rents only this family"))
        for choice in config.gpus:
            price = api.quote(choice.id, config.cloud, config.min_vcpu, config.min_memory_gb)
            state = f"in stock at ${price:.2f}/h" if price else "no stock right now"
            over = " (above its cap)" if price and price > choice.max_usd_per_hour else ""
            print(f"info  {choice.id}: {state}{over}; cap ${choice.max_usd_per_hour:.2f}/h, {choice.processes} processes")
    except Exception as exc:
        check(False, f"RunPod API: {exc}", "export RUNPOD_API_KEY=... (runpod.io > Settings > API Keys)")
    missing = [root for root in config.footage_roots if not Path(root).is_dir()]
    check(not missing, f"footage roots exist: {[str(r) for r in config.footage_roots]}", f"fix [footage] roots; missing {missing}")
    if not missing:
        rows = footage.discover(config.footage_roots, config.extensions, config.exclude)
        size = sum(row[2] for row in rows)
        check(bool(rows), f"{len(rows)} videos, {size / 1e12:.2f} TB under the footage roots", "check [footage] roots and extensions")
        for path, relative, _size in rows[:: max(1, len(rows) // 3)][:3]:
            info = footage.probe(path)
            print(f"info  sample {relative}: {info}")
            check("error" not in info, f"ffprobe reads {relative}", str(info.get("error")))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.output_dir).free
    check(free > 50e9, f"{free / 1e12:.2f} TB free for annotations at {config.output_dir}", "point [output] dir at a larger disk")
    print("\nAll checks passed." if not problems else f"\n{problems} problem(s) to fix.")
    return 1 if problems else 0


def plan(args, config) -> int:
    ledger = ledger_for(config)
    added = scan(config, ledger)
    print(f"{added} new videos registered; reading their lengths (cached for later runs)…")
    probe_all(ledger)
    counts = ledger.counts()
    frames = sum(v[1] for k, v in counts.items() if k in (PENDING, ASSIGNED))
    size = sum(v[2] for k, v in counts.items() if k in (PENDING, ASSIGNED))
    print(f"To annotate: {counts[PENDING][0] + counts[ASSIGNED][0]} videos, {frames:,} frames, {size / 1e12:.2f} TB of video")
    if frames:
        kb = size / frames / 1000
        link = config.upload_limit_mbit * 1e6 / 8
        print(f"Video: {kb:.0f} KB per frame on average; the {config.upload_limit_mbit:.0f} Mbit/s upload limit carries {link / (kb * 1000):.0f} images/s")
        print(f"Annotations: about {frames * OUTPUT_KB_PER_FRAME * 1000 / 1e12:.2f} TB compressed at {config.output_dir}")
        print(f"\nWith {config.pods} pods (rates from docs/speed_test.md, on the lab mix of 2.76 stars per frame; a pilot gives your real rate):")
        for choice in config.gpus:
            if choice.expected_images_per_second:
                rate = choice.expected_images_per_second * config.pods
                capped = min(rate, link / (kb * 1000))
                hours = frames / capped / 3600
                print(f"  {choice.id}: {rate:.0f} images/s, {duration(hours * 3600)}, about ${hours * choice.max_usd_per_hour * config.pods:,.0f}"
                      + ("  (upload-limited)" if capped < rate else ""))
    return 0


def run(args, config) -> int:
    ledger = ledger_for(config)
    added = scan(config, ledger)
    logging.getLogger("fleet").info("%d new videos registered", added)
    if args.pods:
        from dataclasses import replace
        config = replace(config, pods=args.pods)
    if args.max_usd is not None:
        from dataclasses import replace
        config = replace(config, max_total_usd=args.max_usd)
    allowed = load_pilot(config, ledger, args.limit_videos)
    import threading
    threading.Thread(target=probe_all, args=(ledger,), daemon=True).start()
    return Controller(config, ledger, RunPod(), allowed=allowed).run()


def status(args, config) -> int:
    ledger = ledger_for(config)
    counts = ledger.counts()
    print(f"videos: {counts[DONE][0]} done, {counts[ASSIGNED][0]} on pods, {counts[PENDING][0]} pending, {counts[FAILED][0]} failed")
    print(f"frames annotated: {sum(row['frames_done'] or 0 for row in ledger.pods()):,}; estimated spend ${ledger.spent():.2f} of ${config.max_total_usd:.2f}")
    path = config.run_dir / "status.json"
    if path.exists():
        snapshot = json.loads(path.read_text())
        print(f"as of {time.time() - snapshot['updated_epoch']:.0f} s ago: {snapshot['images_per_second']:.1f} images/s, "
              f"${snapshot['usd_per_hour']:.2f}/h, about {duration(snapshot['eta_seconds'])} left, projected total ${snapshot['projected_total_usd']:.2f}")
    for row in ledger.pods(open_only=True):
        print(f"  {row['name']}  {row['gpu']}  {row['state']}  ${row['rate']:.2f}/h  {row['frames_done']:,} frames  {row['detail'] or ''}")
    for row in ledger.failures():
        print(f"  FAILED {row['relative']}: {row['error']}")
    return 0


def report(args, config) -> int:
    """Measured rate and cost per GPU type, and what the rest of the archive would cost."""
    ledger = ledger_for(config)
    groups: dict = {}
    for row in ledger.pods():
        if row["frames_done"] and row["first_frame"] and row["last_frame"] and row["last_frame"] > row["first_frame"]:
            group = groups.setdefault(row["gpu"], [0, 0.0])
            group[0] += row["frames_done"]
            group[1] += row["last_frame"] - row["first_frame"]
    counts = ledger.counts()
    left = counts[PENDING][1] + counts[ASSIGNED][1]
    print(f"{counts[DONE][0]} videos done, {left:,} frames left; these rates include each pod's start-up and video gaps")
    for gpu, (frames, seconds) in groups.items():
        rate = frames / seconds
        hourly = next((c.max_usd_per_hour for c in config.gpus if c.id == gpu), None)
        cost = f"; ${hourly / (rate * 3600) * 1e6:.2f} per million frames at ${hourly:.2f}/h" if hourly else ""
        print(f"  {gpu}: {rate:.1f} images/s per pod over {frames:,} frames{cost}")
        if hourly and left:
            print(f"    rest of the archive: about ${left / (rate * 3600) * hourly:,.0f}; {duration(left / rate / config.pods)} with {config.pods} pods")
    mismatched = ledger._run("SELECT COUNT(*) n FROM videos WHERE state=? AND frames IS NOT NULL AND ABS(frames - frames_done) > 1", (DONE,))[0]["n"]
    if mismatched:
        print(f"{mismatched} videos gave a different frame count than their container reports; see fleet.log")
    sample = sorted(Path(config.output_dir).rglob("*.jsonl.zst"))[:: max(1, args.sample_every)][:40]
    stars, frames = 0, 0
    for path in sample:
        output = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout
        for line in output.splitlines():
            stars += len(json.loads(line)["stars"])
            frames += 1
    if frames:
        print(f"Stars per frame in {len(sample)} sampled videos: {stars / frames:.2f} (the speed test used 2.76; more stars means slower)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backcast", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helper in (("doctor", "check this machine, the RunPod key and the footage"), ("plan", "count the work and estimate cost and time"),
                         ("run", "rent pods and annotate; safe to rerun, resumes where it stopped"), ("status", "progress from another terminal"),
                         ("report", "measured rate and cost per GPU type"), ("stop", "let the pods finish what they hold, then end"),
                         ("terminate", "end every pod of this job right now"), ("retry-failed", "put failed videos back in the queue")):
        command = sub.add_parser(name, help=helper)
        command.add_argument("--config", type=Path, default=Path("fleet.toml"))
        if name == "run":
            command.add_argument("--limit-videos", type=int, help="pilot: annotate only this many videos, evenly spaced through the archive")
            command.add_argument("--pods", type=int, help="override [fleet] pods")
            command.add_argument("--max-usd", type=float, help="override [fleet] max_total_usd")
        if name == "report":
            command.add_argument("--sample-every", type=int, default=25)
    args = parser.parse_args(argv)
    config = load(args.config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(config.run_dir / "fleet.log")])
    logging.Formatter.converter = time.gmtime
    if args.command == "stop":
        (config.run_dir / "STOP").write_text("stop\n")
        print("The running fleet will finish what is on its pods, then end.")
        return 0
    if args.command == "terminate":
        print(f"Terminated {terminate_open_pods(config, ledger_for(config), RunPod())} pods.")
        return 0
    if args.command == "retry-failed":
        print(f"{ledger_for(config).retry_failed()} videos back in the queue.")
        return 0
    return {"doctor": doctor, "plan": plan, "run": run, "status": status, "report": report}[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
