"""Run commands and move files on a pod. LocalRemote stands in for SSH in tests and rehearsals."""
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


class SshRemote:
    def __init__(self, host: str, port: int, key: Path, known_hosts: Path, control_dir: Path, user: str = "root") -> None:
        self.host, self.port = host, port
        self.common = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                       "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}", "-i", str(key), "-p", str(port)]
        # Short commands share one connection; transfers open their own, which goes faster on long paths.
        self.multiplex = ["-o", "ControlMaster=auto", "-o", f"ControlPath={control_dir}/%C", "-o", "ControlPersist=300"]
        self.target = f"{user}@{host}"

    def run(self, command: str, timeout: float = 120, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *self.common, *self.multiplex, self.target, command], input=stdin,
                              capture_output=True, text=True, timeout=timeout)

    def put(self, local: Path | str, remote: str, *, bwlimit_kbps: int | None = None, timeout: float = 7200, recursive: bool = False) -> subprocess.CompletedProcess:
        rsh = " ".join(shlex.quote(item) for item in ["ssh", *self.common])
        command = ["rsync", "-e", rsh, "--whole-file"]
        if recursive:
            command += ["-a", "--delete", "--exclude", "__pycache__", "--exclude", ".git"]
        if bwlimit_kbps:
            command.append(f"--bwlimit={int(bwlimit_kbps)}")
        return subprocess.run([*command, str(local), f"{self.target}:{remote}"], capture_output=True, text=True, timeout=timeout)

    def get_stream(self, command: str, local: Path, timeout: float = 3600) -> subprocess.CompletedProcess:
        with Path(local).open("wb") as sink:
            return subprocess.run(["ssh", *self.common, self.target, command], stdout=sink, stderr=subprocess.PIPE, timeout=timeout)


class LocalRemote:
    """The pod's files live in a local directory: commands use the same absolute paths."""

    host = "localhost"
    port = 0

    def run(self, command: str, timeout: float = 120, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", command], input=stdin, capture_output=True, text=True, timeout=timeout)

    def put(self, local, remote: str, *, bwlimit_kbps=None, timeout: float = 7200, recursive: bool = False) -> subprocess.CompletedProcess:
        command = ["rsync", "--whole-file"] + (["-a", "--delete", "--exclude", "__pycache__", "--exclude", ".git"] if recursive else [])
        return subprocess.run([*command, str(local), remote], capture_output=True, text=True, timeout=timeout)

    def get_stream(self, command: str, local, timeout: float = 3600) -> subprocess.CompletedProcess:
        with Path(local).open("wb") as sink:
            return subprocess.run(["bash", "-c", command], stdout=sink, stderr=subprocess.PIPE, timeout=timeout)
