#!/usr/bin/env python3
"""Print one JSON line describing this pod's queue, outputs and GPU. Read-only. Usage: pod_status.py ROOT"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
queue = root / "queue"


def lines(path):
    with path.open("rb") as stream:
        return sum(block.count(b"\n") for block in iter(lambda: stream.read(1 << 20), b""))


def names(folder):
    return sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []


status = {"ready": names(queue / "ready"), "failed": names(queue / "failed"),
          "working": sorted(p.name for p in (queue / "work").glob("*/*")),
          "partial": {}, "finished": {}}
for path in (root / "out").glob("*.jsonl.part"):
    status["partial"][path.name[: -len(".jsonl.part")]] = lines(path)
for path in (root / "out").glob("*.jsonl"):
    status["finished"][path.name[: -len(".jsonl")]] = lines(path)
try:
    pid = int((root / "service.pid").read_text())
    os.kill(pid, 0)
    status["service"] = (Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z")
except (OSError, ValueError):
    status["service"] = False
status["workers_ready"] = len(list((root / "out").glob("ready.*")))
status["workers_alive"] = 0
for folder in Path("/proc").iterdir():
    if folder.name.isdigit():
        try:
            status["workers_alive"] += b"--worker-index" in (folder / "cmdline").read_bytes()
        except OSError:
            pass
status["disk_free_gb"] = shutil.disk_usage(root).free / 1e9
try:
    row = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True).stdout
    name, used, total, util, power = [item.strip() for item in row.split(",")]
    status["gpu"] = {"name": name, "memory_used_mib": float(used), "memory_total_mib": float(total), "utilization_percent": float(util),
                     "power_watts": None if "N/A" in power else float(power)}
except (OSError, ValueError, subprocess.SubprocessError):
    status["gpu"] = None
print(json.dumps(status))
