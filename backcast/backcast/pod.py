"""One pod's life: set it up, keep its queue full, bring finished annotations home, then shut it down."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

from . import ledger as states

FILES = Path(__file__).resolve().parent / "pod_files"
INFERENCE = Path(__file__).resolve().parent.parent / "inference"
SYSTEM_PACKAGES = "rsync zstd libgl1 libglib2.0-0"
POLL_SECONDS = 20.0
BOOT_SECONDS = 25 * 60
INSTALL_SECONDS = 30 * 60
UNREACHABLE_SECONDS = 6 * 60
EXPIRY_MARGIN_SECONDS = 45 * 60
MAX_SERVICE_RESTARTS = 2
SERVICE_RESTART_WINDOW_SECONDS = 30 * 60
MIN_DISK_GB = 8.0
MAX_FETCH_TRIES = 4
WORKER_START_SECONDS = 90.0


def remote_name(video_id: int, relative: str) -> str:
    """A unique, shell-safe file name for one video on the pod; it starts with the ledger id."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(relative).stem)[:80]
    return f"{video_id:08d}__{stem}"


def video_id_of(name: str) -> int:
    return int(name.split("__", 1)[0])


class PodFailure(RuntimeError):
    pass


class Pod(threading.Thread):
    """Runs one pod to its end on its own thread; the controller only starts, counts and stops these."""

    def __init__(self, *, name, pod_id, gpu, processes, created, config, ledger, api, connect, sink, work_left, allowed=None,
                 poll_seconds=POLL_SECONDS, adopted=False) -> None:
        super().__init__(name=name, daemon=True)
        self.pod_name, self.pod_id, self.gpu, self.processes, self.created = name, pod_id, gpu, processes, created
        self.config, self.ledger, self.api, self.connect, self.sink = config, ledger, api, connect, sink
        self.work_left = work_left
        self.allowed, self.poll_seconds, self.adopted = allowed, poll_seconds, adopted
        self.log = logging.getLogger(f"pod.{name}")
        self.root = config.pod_root
        self.remote = None
        self.stop_feeding = threading.Event()
        self.kill_now = threading.Event()
        self.closed = threading.Event()
        self.state = "starting"
        self.detail = ""
        self.inflight = 0
        self.lock = threading.Lock()
        self.uploads = ThreadPoolExecutor(max_workers=config.uploads_per_pod, thread_name_prefix=f"{name}-up")
        self.frames_seen = 0
        self.last_progress = time.time()
        self.service_restarts = 0
        self.last_restart = 0.0
        self.status: dict = {}
        self.rate = 0.0
        self._last_visible = -1
        self.fetch_tries: dict[str, int] = {}
        self.service_started = 0.0
        self._rate_mark: tuple[float, int] | None = None
        self.gpu_name = gpu

    # ------------------------------------------------------------ helpers
    def set_state(self, state: str, detail: str = "") -> None:
        self.state, self.detail = state, detail
        self.ledger.update_pod(self.pod_name, state=state, detail=detail[:300])
        self.log.info("%s %s", state, detail)

    def run_checked(self, command: str, timeout: float = 120) -> str:
        result = self.remote.run(command, timeout=timeout)
        if result.returncode:
            raise PodFailure(f"`{command[:60]}` exit {result.returncode}: {result.stderr.strip()[-400:]}")
        return result.stdout

    def elapsed(self) -> float:
        return time.time() - self.created

    # ------------------------------------------------------------- set up
    def wait_for_ssh(self) -> None:
        self.set_state("booting")
        deadline = time.time() + BOOT_SECONDS
        while time.time() < deadline and not self.kill_now.is_set():
            try:
                self.remote = self.connect(self)
                if self.remote is not None and self.remote.run("true", timeout=40).returncode == 0:
                    return
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                self.log.info("not reachable yet: %s", exc)
            self.kill_now.wait(10)
        raise PodFailure("no SSH before the boot deadline")

    def install(self) -> None:
        if self.remote.run(f"test -f {self.root}/INSTALLED").returncode == 0:
            return
        self.set_state("installing")
        apt = ("command -v rsync >/dev/null && command -v zstd >/dev/null && ldconfig -p | grep -q libgthread-2.0 || "
               f"(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {SYSTEM_PACKAGES} >/dev/null 2>&1)")
        self.run_checked(apt, timeout=600)
        self.run_checked(f"mkdir -p {self.root}/code {self.root}/queue/ready {self.root}/queue/incoming {self.root}/out {self.root}/logs")
        for local, name in ((FILES, "pod_files"), (INFERENCE, "inference")):
            if not Path(local).is_dir():
                raise PodFailure(f"missing local directory {local}")
            sent = self.remote.put(f"{local}/", f"{self.root}/code/{name}/", recursive=True, timeout=900)
            if sent.returncode:
                raise PodFailure(f"code upload failed: {sent.stderr.strip()[-300:]}")
        command = f"bash {self.root}/code/pod_files/install.sh {self.root} {shlex.quote(self.config.models_url)} > {self.root}/logs/install.log 2>&1"
        started = time.time()
        result = self.remote.run(command, timeout=INSTALL_SECONDS)
        if result.returncode:
            tail = self.remote.run(f"tail -c 800 {self.root}/logs/install.log").stdout
            raise PodFailure(f"install failed: {tail}")
        self.log.info("installed in %.0f s", time.time() - started)

    def start_service(self) -> None:
        self.run_checked(f"bash {self.root}/code/pod_files/serve.sh {self.root} {self.processes}")
        self.service_started = time.time()

    # ------------------------------------------------------------ polling
    def poll(self) -> dict | None:
        try:
            result = self.remote.run(f"python3 {self.root}/code/pod_files/pod_status.py {self.root}", timeout=90)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode:
            return None
        try:
            return json.loads(result.stdout)
        except ValueError:
            return None

    def track_progress(self, status: dict) -> None:
        partial = sum(status["partial"].values())
        finished = sum(status["finished"].values())
        total = self.frames_seen + partial + finished
        now = time.time()
        mark = self._rate_mark
        if mark is None:
            self._rate_mark = (now, total)
        elif now - mark[0] >= 60:
            self.rate = max(0.0, total - mark[1]) / (now - mark[0])
            self._rate_mark = (now, total)
        if partial + finished != self._last_visible:
            self._last_visible = partial + finished
            self.last_progress = now

    # ------------------------------------------------------------ feeding
    def feed(self, status: dict) -> None:
        if self.stop_feeding.is_set() or status["disk_free_gb"] < MIN_DISK_GB:
            return
        waiting = len(status["ready"])
        while True:
            with self.lock:
                if waiting + self.inflight >= self.config.ready_videos or self.inflight >= self.config.uploads_per_pod:
                    return
            row = self.ledger.claim(self.pod_name, self.allowed)
            if row is None:
                return
            with self.lock:
                self.inflight += 1
            self.uploads.submit(self.upload, row)

    def upload(self, row) -> None:
        name = remote_name(row["id"], row["relative"]) + Path(row["path"]).suffix.lower()
        limit = self.config.upload_limit_mbit * 1000 / 8 / max(1, self.config.pods * self.config.uploads_per_pod)
        try:
            sent = self.remote.put(row["path"], f"{self.root}/queue/incoming/{name}.part", bwlimit_kbps=int(limit))
            if sent.returncode:
                raise PodFailure(sent.stderr.strip()[-200:] or "rsync failed")
            self.run_checked(f"mv {self.root}/queue/incoming/{shlex.quote(name)}.part {self.root}/queue/ready/{shlex.quote(name)}")
        except (PodFailure, OSError, subprocess.SubprocessError) as exc:
            state = self.ledger.requeue(row["id"], f"upload: {exc}", self.config.max_attempts)
            self.log.warning("upload failed for %s (now %s): %s", row["relative"], state, exc)
        finally:
            with self.lock:
                self.inflight -= 1

    # --------------------------------------------------------- collecting
    def collect(self, status: dict) -> None:
        for name, frames in status["finished"].items():
            try:
                self.fetch(name, frames)
                self.fetch_tries.pop(name, None)
            except (PodFailure, OSError, subprocess.SubprocessError) as exc:
                # Usually a network blip. The annotations are safe on the pod until a verified copy is home.
                self.fetch_tries[name] = self.fetch_tries.get(name, 0) + 1
                self.log.warning("fetch %s failed (try %d): %s", name, self.fetch_tries[name], exc)
                if self.fetch_tries[name] >= MAX_FETCH_TRIES:
                    raise PodFailure(f"could not fetch {name}: {exc}") from None
        for name in status["failed"]:
            video = video_id_of(name)
            tail = self.remote.run(f"tail -n 3 {self.root}/out/worker.*.log 2>/dev/null | tail -c 300").stdout
            self.ledger.fail(video, f"could not be decoded on {self.gpu_name}: {tail.strip()[-200:]}")
            self.log.warning("video %s failed to decode", name)
            self.run_checked(f"rm -f {self.root}/queue/failed/{shlex.quote(name)} {self.root}/out/{shlex.quote(name.rsplit('.', 1)[0])}.jsonl.part")

    def fetch(self, name: str, frames: int) -> None:
        video = video_id_of(name)
        row = self.ledger.get(video)
        remote = f"{self.root}/out/{name}.jsonl"
        if row is None or row["state"] == states.DONE:
            self.run_checked(f"rm -f {remote}")
            return
        final = self.config.output_dir / (row["relative"] + ".jsonl.zst")
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + f".{self.pod_name}.part")
        result = self.remote.get_stream(f"zstd -3 -q -c {remote}", partial)
        if result.returncode:
            partial.unlink(missing_ok=True)
            raise PodFailure(f"fetch failed for {name}: {result.stderr.decode(errors='replace')[-200:]}")
        counted = count_lines(partial)
        if counted != frames:
            partial.unlink(missing_ok=True)
            raise PodFailure(f"{name}: fetched {counted} records of {frames}")
        os.replace(partial, final)
        self.ledger.finish(video, frames, self.gpu_name)
        self.sink({"relative": row["relative"], "frames": frames, "expected_frames": row["frames"], "gpu": self.gpu_name,
                   "pod": self.pod_name, "finished_epoch": time.time(), "bytes": final.stat().st_size})
        if row["frames"] and abs(row["frames"] - frames) > 1:
            self.log.warning("%s: annotated %d frames; its container says %d", row["relative"], frames, row["frames"])
        self.frames_seen += frames
        self.ledger.update_pod(self.pod_name, frames_done=self.frames_seen, last_frame=time.time())
        self.run_checked(f"rm -f {remote}")

    # ----------------------------------------------------- service health
    def restart_service(self, why: str) -> None:
        # Workers fault very rarely (inference/.../detect_ahead.py describes it), and over days a pod will see some.
        # Only restarts close together mean something is wrong with this pod.
        if time.time() - self.last_restart > SERVICE_RESTART_WINDOW_SECONDS:
            self.service_restarts = 0
        self.last_restart = time.time()
        self.service_restarts += 1
        if self.service_restarts > MAX_SERVICE_RESTARTS:
            raise PodFailure(f"service keeps failing: {why}")
        tail = self.remote.run(f"tail -c 600 {self.root}/out/worker.0.log; tail -c 300 {self.root}/logs/service.log").stdout
        self.log.warning("restarting the service (%s); log tail: %s", why, tail[-500:])
        # A video in progress goes back to the queue; its partial records are worthless.
        # The bracket keeps this shell's own command line from matching the pattern, and the
        # queue path limits it to this job's service.
        self.remote.run(f"pkill -f '[d]eepstarnoodles_inference serve-videos --queue {self.root}/queue'; sleep 2; "
                        f"for f in {self.root}/queue/work/*/*; do [ -e \"$f\" ] && mv \"$f\" {self.root}/queue/ready/; done; "
                        f"rm -f {self.root}/out/*.part {self.root}/out/ready.* {self.root}/queue/STOP")
        self.start_service()
        self.last_progress = time.time()

    def reconcile(self, status: dict) -> None:
        """After adoption: videos the ledger thinks are here but the pod does not have go back to the queue."""
        present = {video_id_of(name) for name in [*status["ready"], *status["working"], *status["failed"], *status["finished"]]}
        for row in self.ledger.assigned_to(self.pod_name):
            if row["id"] not in present:
                self.ledger.requeue(row["id"], "lost while the controller was down", self.config.max_attempts)

    # --------------------------------------------------------------- main
    def run(self) -> None:
        reason = "finished"
        try:
            self.wait_for_ssh()
            if self.adopted:
                status = self.poll()
                if status is None or self.remote.run(f"test -f {self.root}/INSTALLED").returncode:
                    raise PodFailure("adopted pod is not ready")
                self.remote.run(f"rm -f {self.root}/queue/incoming/*")
                self.reconcile(status)
                if not status["service"]:
                    self.restart_service("service not running after adoption")
            else:
                self.install()
                self.start_service()
            self.set_state("serving", f"{self.processes} processes on {self.gpu}")
            self.ledger.update_pod(self.pod_name, first_frame=time.time())
            self.serve()
        except PodFailure as exc:
            reason = f"failed: {exc}"
            self.log.error("%s", reason)
        except Exception as exc:  # A bug or surprise on one pod must not leave it billing.
            reason = f"failed: {type(exc).__name__}: {exc}"
            self.log.exception("unexpected error")
        finally:
            self.shutdown(reason)

    def serve(self) -> None:
        unreachable_since = None
        while not self.kill_now.is_set():
            status = self.poll()
            if status is None:
                unreachable_since = unreachable_since or time.time()
                if time.time() - unreachable_since > UNREACHABLE_SECONDS:
                    raise PodFailure("unreachable")
                self.kill_now.wait(self.poll_seconds)
                continue
            unreachable_since = None
            self.status = status
            if status["gpu"]:
                self.gpu_name = status["gpu"]["name"]
            self.track_progress(status)
            if self.elapsed() > self.config.lease_hours * 3600 - EXPIRY_MARGIN_SECONDS:
                self.stop_feeding.set()  # A replacement takes over before RunPod ends this pod.
            try:
                self.collect(status)
                self.feed(status)
            except subprocess.SubprocessError as exc:
                self.log.warning("network trouble: %s", exc)
                self.kill_now.wait(self.poll_seconds)
                continue
            with self.lock:
                idle = not (self.inflight or status["ready"] or status["working"] or status["finished"] or status["partial"])
            if idle and self.stop_feeding.is_set():
                return
            if idle and not self.work_left():
                return
            workers_due = time.time() - self.service_started > WORKER_START_SECONDS
            if not status["service"] or (workers_due and status["workers_alive"] < self.processes):
                self.restart_service(f"{status['workers_alive']} of {self.processes} workers running")
            elif (status["working"] or status["ready"]) and time.time() - self.last_progress > self.config.stall_minutes * 60:
                self.restart_service(f"no new frames for {self.config.stall_minutes:.0f} minutes")
            self.kill_now.wait(self.poll_seconds)

    def shutdown(self, reason: str) -> None:
        self.stop_feeding.set()
        self.uploads.shutdown(wait=False, cancel_futures=True)
        released = self.ledger.release_pod(self.pod_name, f"pod ended: {reason}", self.config.max_attempts)
        terminated = False
        for _ in range(5):
            try:
                self.api.terminate(self.pod_id)
                terminated = True
                break
            except Exception as exc:  # Keep trying: a pod left running keeps billing.
                self.log.error("terminate failed: %s", exc)
                try:
                    if self.api.pod(self.pod_id) is None:  # RunPod has already removed it.
                        terminated = True
                        break
                except Exception:
                    pass
                time.sleep(5)
        self.ledger.update_pod(self.pod_name, ended=time.time() if terminated else None,
                               state="ended" if terminated else "terminate_failed", detail=reason[:300])
        if not terminated:
            self.log.error("COULD NOT TERMINATE POD %s: it is still billing; run `backcast terminate` or remove it at runpod.io", self.pod_id)
        self.log.info("ended (%s); %d videos returned to the queue; frames %d", reason, released, self.frames_seen)
        self.closed.set()


def count_lines(path: Path) -> int:
    """Records in a compressed file; decoding it also proves the file is intact."""
    with subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        total = sum(block.count(b"\n") for block in iter(lambda: process.stdout.read(1 << 20), b""))
    return -1 if process.returncode else total
