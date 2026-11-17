"""The fleet: keep the configured number of pods working until the archive is done or the budget is spent."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import threading
import time

from . import footage
from .ledger import ASSIGNED, DONE, FAILED, PENDING, Ledger
from .pod import Pod
from .remote import SshRemote
from .runpod_api import SupplyError, ssh_endpoint

log = logging.getLogger("fleet")
STOP_FEEDING_AT = 0.97  # of the budget; running videos can still finish and come home
MIN_OUTPUT_FREE_BYTES = 20e9
RETRY_STOCK_SECONDS = 120.0
# Pods ending in a row without one finished video mean something systematic is wrong:
# stop renting and let a person look, rather than paying for each new pod to fail the same way.
MAX_FUTILE_PODS = 3
PILOT_FILE = "pilot_videos.json"


def ledger_for(config) -> Ledger:
    return Ledger(config.run_dir / "state.sqlite")


def ensure_key(run_dir: Path) -> tuple[Path, str]:
    """A key used only by this job, so the lab user's own SSH keys never reach a pod."""
    key = run_dir / "id_ed25519"
    if not key.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "deepstarnoodles-backcast", "-f", str(key)], check=True)
    return key, key.with_suffix(".pub").read_text().strip()


def scan(config, ledger: Ledger) -> int:
    rows = footage.discover(config.footage_roots, config.extensions, config.exclude)
    return ledger.add_videos(rows)


def probe_all(ledger: Ledger, limit=None) -> int:
    paths = ledger.unprobed(limit)
    for start in range(0, len(paths), 64):
        chunk = paths[start:start + 64]
        for path, info in footage.probe_many(chunk).items():
            ledger.set_probe(path, info)
    return len(paths)


class Controller:
    def __init__(self, config, ledger, api, *, connect=None, allowed=None, poll_seconds=None, retry_stock_seconds=RETRY_STOCK_SECONDS) -> None:
        self.config, self.ledger, self.api = config, ledger, api
        self.allowed = allowed
        self.pods: list[Pod] = []
        self.draining = threading.Event()
        self.killing = threading.Event()
        self.poll_seconds = poll_seconds
        self.retry_stock_seconds = retry_stock_seconds
        self.next_stock_try = 0.0
        self.tripped = False
        self.started_epoch = time.time()
        self.manifest_lock = threading.Lock()
        self.public_key = ""
        self.connect = connect or self.connect_ssh
        config.run_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self.control_dir = Path(f"/tmp/backcast-{secrets.token_hex(3)}")

    # ------------------------------------------------------------- wiring
    def connect_ssh(self, pod: Pod):
        info = self.api.pod(pod.pod_id)
        endpoint = ssh_endpoint(info)
        if endpoint is None:
            return None
        self.control_dir.mkdir(mode=0o700, exist_ok=True)
        return SshRemote(endpoint[0], endpoint[1], self.config.run_dir / "id_ed25519", self.config.run_dir / "known_hosts", self.control_dir)

    def record_output(self, row: dict) -> None:
        with self.manifest_lock, (self.config.output_dir / "_manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def work_left(self) -> bool:
        return self.ledger.pending_count(self.allowed) > 0

    def make_pod(self, *, name, pod_id, gpu, processes, created, adopted=False) -> Pod:
        options = {"poll_seconds": self.poll_seconds} if self.poll_seconds is not None else {}
        pod = Pod(name=name, pod_id=pod_id, gpu=gpu, processes=processes, created=created, config=self.config, ledger=self.ledger,
                  api=self.api, connect=self.connect, sink=self.record_output, work_left=self.work_left, allowed=self.allowed,
                  adopted=adopted, **options)
        self.pods.append(pod)
        pod.start()
        return pod

    # --------------------------------------------------------------- pods
    def alive(self) -> list[Pod]:
        return [pod for pod in self.pods if not pod.closed.is_set()]

    def adopt_open_pods(self) -> None:
        """Pick up pods that survived a restart of this controller; close what RunPod no longer has."""
        for row in self.ledger.pods(open_only=True):
            info = None
            try:
                info = self.api.pod(row["pod_id"])
            except Exception as exc:
                log.warning("could not look up pod %s: %s", row["name"], exc)
            if not info or info.get("desiredStatus") != "RUNNING":
                self.ledger.update_pod(row["name"], ended=time.time(), state="ended", detail="gone at restart")
                self.ledger.release_pod(row["name"], "pod gone at restart", self.config.max_attempts)
                continue
            if row["gpu"] not in {choice.id for choice in self.config.gpus}:
                self.retire(row)
                continue
            log.info("adopting pod %s", row["name"])
            self.make_pod(name=row["name"], pod_id=row["pod_id"], gpu=row["gpu"], processes=row["processes"], created=row["created"], adopted=True)

    def retire(self, row) -> None:
        """End a surviving pod whose GPU fleet.toml no longer allows, so a changed `gpu_family` holds from now on."""
        log.warning("terminating pod %s: fleet.toml no longer allows %s; its unfinished videos go back in the queue", row["name"], row["gpu"])
        try:
            self.api.terminate(row["pod_id"])
        except Exception as exc:
            log.error("COULD NOT TERMINATE POD %s: %s. It is still billing; run `backcast terminate` or remove it at runpod.io", row["pod_id"], exc)
            return
        self.ledger.update_pod(row["name"], ended=time.time(), state="ended", detail="GPU no longer in fleet.toml")
        self.ledger.release_pod(row["name"], "pod's GPU no longer in fleet.toml", self.config.max_attempts)

    def start_pod(self) -> bool:
        """Rent one pod: the first GPU in the config's order that is in stock at its price cap."""
        for choice in self.config.gpus:
            if choice.max_pods is not None and sum(pod.gpu == choice.id for pod in self.alive()) >= choice.max_pods:
                continue
            try:
                price = self.api.quote(choice.id, self.config.cloud, self.config.min_vcpu, self.config.min_memory_gb)
                if price is None or price > choice.max_usd_per_hour:
                    continue
                now = time.time()
                name = f"backcast-{int(now):x}-{secrets.token_hex(2)}"
                stop = now + self.config.lease_hours * 3600
                created = self.api.deploy(name=name, gpu=choice.id, cloud=self.config.cloud, image=self.config.image,
                                          container_gb=self.config.container_gb, min_vcpu=self.config.min_vcpu,
                                          min_memory_gb=self.config.min_memory_gb, public_key=self.public_key,
                                          stop_epoch=stop, terminate_epoch=stop + 15 * 60)
            except SupplyError:
                continue
            except Exception as exc:
                log.warning("could not rent %s: %s", choice.id, exc)
                continue
            rate = float(created.get("costPerHr") or price)
            self.ledger.add_pod(name, created["id"], choice.id, choice.processes, rate, now)
            if rate > choice.max_usd_per_hour:
                log.warning("%s came back at $%.2f/h, above its cap; terminating", choice.id, rate)
                self.api.terminate(created["id"])
                self.ledger.update_pod(name, ended=time.time(), state="ended", detail="above price cap")
                continue
            log.info("rented %s (%s) at $%.2f/h", name, choice.id, rate)
            self.make_pod(name=name, pod_id=created["id"], gpu=choice.id, processes=choice.processes, created=now)
            return True
        return False

    def futile_pods(self) -> list:
        """The most recent run of ended pods that finished no video."""
        run = []
        for row in reversed(self.ledger.pods()):
            # Only this run counts: a rerun after fixing the cause starts clean.
            if row["ended"] is None or row["created"] < self.started_epoch:
                continue
            if row["frames_done"]:
                break
            if str(row["detail"] or "").startswith("failed"):
                run.append(row)
        return run

    def manage_fleet(self) -> None:
        now = time.time()
        if self.draining.is_set() or not self.work_left() or now < self.next_stock_try:
            return
        futile = self.futile_pods()
        if len(futile) >= MAX_FUTILE_PODS:
            self.tripped = True
            log.error("%d pods in a row failed without finishing a video; not renting any more. The last one said: %s",
                      len(futile), futile[0]["detail"])
            self.drain()
            return
        active = [pod for pod in self.alive() if not pod.stop_feeding.is_set()]
        # One rental per pass, so a bad type cannot burn through several pods before the next look.
        if len(active) < self.config.pods and not self.start_pod():
            log.info("no GPU in stock within its price cap; retrying in %.0f s", self.retry_stock_seconds)
            self.next_stock_try = now + self.retry_stock_seconds

    def enforce_budget(self) -> float:
        spent = self.ledger.spent()
        if spent >= self.config.max_total_usd:
            if not self.killing.is_set():
                log.error("budget of $%.2f reached; terminating every pod", self.config.max_total_usd)
            self.kill_all()
        elif spent >= self.config.max_total_usd * STOP_FEEDING_AT and not self.draining.is_set():
            log.warning("%.0f%% of the budget used; finishing what is on the pods", 100 * spent / self.config.max_total_usd)
            self.drain()
        return spent

    def check_output_disk(self) -> None:
        if not self.draining.is_set() and shutil.disk_usage(self.config.output_dir).free < MIN_OUTPUT_FREE_BYTES:
            log.error("under %.0f GB free at %s: stopping; free some space and rerun", MIN_OUTPUT_FREE_BYTES / 1e9, self.config.output_dir)
            self.drain()

    def drain(self) -> None:
        self.draining.set()
        for pod in self.pods:
            pod.stop_feeding.set()

    def kill_all(self) -> None:
        self.killing.set()
        self.drain()
        for pod in self.pods:
            pod.kill_now.set()

    # ------------------------------------------------------------- status
    def write_status(self) -> dict:
        counts = self.ledger.counts()
        rate = sum(pod.rate for pod in self.alive())
        hourly = sum(row["rate"] or 0 for row in self.ledger.pods(open_only=True))
        known = counts[PENDING][1] + counts[ASSIGNED][1]
        eta = known / rate if rate > 0 and known else None
        spent = self.ledger.spent()
        status = {"updated_epoch": time.time(), "videos": {k: v[0] for k, v in counts.items()},
                  "frames_done": counts[DONE][1], "frames_left_known": known, "images_per_second": rate, "eta_seconds": eta,
                  "spent_usd": spent, "budget_usd": self.config.max_total_usd, "usd_per_hour": hourly,
                  "projected_total_usd": spent + (eta or 0) / 3600 * hourly,
                  "pods": [{"name": pod.pod_name, "gpu": pod.gpu, "state": pod.state, "images_per_second": pod.rate,
                            "frames_done": pod.frames_seen, "ready": len(pod.status.get("ready", [])), "working": len(pod.status.get("working", [])),
                            "gpu_utilization": (pod.status.get("gpu") or {}).get("utilization_percent"),
                            "power_watts": (pod.status.get("gpu") or {}).get("power_watts")} for pod in self.alive()]}
        path = self.config.run_dir / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=1) + "\n")
        temporary.replace(path)
        return status

    # --------------------------------------------------------------- main
    def run(self) -> int:
        _key, self.public_key = ensure_key(self.config.run_dir)
        stop_file = self.config.run_dir / "STOP"
        stop_file.unlink(missing_ok=True)
        previous = {}

        def on_signal(number, _frame):
            if self.killing.is_set():
                log.error("second signal: exiting now; run `backcast terminate` if any pod is still listed")
                raise SystemExit(130)
            log.warning("signal %d: terminating every pod; unfinished videos stay in the queue for the next run", number)
            self.kill_all()

        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                previous[number] = signal.signal(number, on_signal)
        self.adopt_open_pods()
        last_report = 0.0
        tick = min(5.0, self.poll_seconds or 5.0)
        try:
            while True:
                if stop_file.exists() and not self.draining.is_set():
                    log.info("stop requested: finishing what is on the pods")
                    self.drain()
                self.enforce_budget()
                self.check_output_disk()
                self.manage_fleet()
                now = time.time()
                if now - last_report >= 60 or self.poll_seconds:
                    status = self.write_status()
                    if now - last_report >= 60:
                        last_report = now
                        log.info("done %d / left %d videos · %.1f images/s · $%.2f of $%.0f · %d pods · failed %d",
                                 status["videos"][DONE], status["videos"][PENDING] + status["videos"][ASSIGNED], status["images_per_second"],
                                 status["spent_usd"], status["budget_usd"], len(status["pods"]), status["videos"][FAILED])
                if not self.alive() and (self.draining.is_set() or not self.work_left()):
                    break
                time.sleep(tick)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            shutil.rmtree(self.control_dir, ignore_errors=True)
        status = self.write_status()
        counts = self.ledger.counts()
        log.info("fleet stopped: %d done, %d pending, %d failed; about $%.2f", counts[DONE][0], counts[PENDING][0], counts[FAILED][0], status["spent_usd"])
        if self.tripped:
            log.error("stopped because pods keep failing: see %s, fix the cause, then rerun", self.config.run_dir / "fleet.log")
            return 4
        return 0 if not self.work_left() and not counts[FAILED][0] else 3


def terminate_open_pods(config, ledger: Ledger, api) -> int:
    """Stop paying right now for everything this job has open. Unfinished videos go back in the queue."""
    closed = 0
    for row in ledger.pods(open_only=True):
        api.terminate(row["pod_id"])
        ledger.update_pod(row["name"], ended=time.time(), state="ended", detail="terminated by hand")
        ledger.release_pod(row["name"], "terminated by hand", config.max_attempts)
        closed += 1
    return closed


def load_pilot(config, ledger: Ledger, count: int | None) -> set[int] | None:
    """The evenly spaced videos of a pilot, kept on disk so that resuming it keeps the same sample."""
    path = config.run_dir / PILOT_FILE
    if count is None:
        return None
    if path.exists():
        chosen = json.loads(path.read_text())
        if len(chosen) == count:
            return set(chosen)
    chosen = ledger.limit_to(count)
    path.write_text(json.dumps(chosen) + "\n")
    return set(chosen)
