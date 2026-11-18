"""The fleet's bookkeeping, end to end, with fake pods: no cloud, no GPU, no money."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backcast import config as config_module, controller, footage, ledger as ledger_module, pod as pod_module  # noqa: E402
from backcast.controller import Controller  # noqa: E402
from backcast.ledger import DONE, FAILED, PENDING, Ledger  # noqa: E402
from backcast.remote import LocalRemote  # noqa: E402
from backcast.runpod_api import SupplyError  # noqa: E402

# Stands in for serve-videos: a "video" is a text file holding its frame count, or a word that triggers a behavior.
FAKE_SERVICE = r'''#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
ready, out = root / "queue" / "ready", root / "out"
work = root / "queue" / "work" / "0"
work.mkdir(parents=True, exist_ok=True)
(out / "ready.0").write_text("up")
while not (root / "queue" / "STOP").exists():
    names = sorted(ready.iterdir())
    if not names:
        time.sleep(0.02)
        continue
    video = work / names[0].name
    os.rename(names[0], video)
    text = video.read_text().strip()
    if text == "corrupt":
        (root / "queue" / "failed").mkdir(exist_ok=True)
        os.rename(video, root / "queue" / "failed" / video.name)
        continue
    if text == "crash" and not (root / "crashed").exists():
        (root / "crashed").write_text("x")
        os._exit(1)
    name = video.name.rsplit(".", 1)[0]
    frames = 3 if text == "crash" else int(text)
    part = out / f"{name}.jsonl.part"
    with part.open("w") as sink:
        for index in range(frames):
            sink.write('{"frame": %d, "stars": [1, 2]}\n' % index)
    os.rename(part, out / f"{name}.jsonl")
    video.unlink()
'''

FAKE_SERVE = '''#!/bin/bash
ROOT=$1
nohup setsid python3 "$ROOT/code/pod_files/fake_service.py" "$ROOT" >> "$ROOT/logs/service.log" 2>&1 < /dev/null &
echo $! > "$ROOT/service.pid"
'''


class FakeApi:
    """RunPod as far as the fleet can tell; pods are directories."""

    def __init__(self, stock):
        self.stock = dict(stock)  # gpu -> price, or None for no stock
        self.deployed, self.terminated = [], []
        self.deploy_errors = 0
        self.lock = threading.Lock()

    def quote(self, gpu, cloud, vcpu, memory):
        return self.stock.get(gpu)

    def deploy(self, *, name, gpu, **_):
        with self.lock:
            if self.deploy_errors:
                self.deploy_errors -= 1
                raise SupplyError("no longer any instances")
            pod_id = f"id{len(self.deployed)}"
            self.deployed.append((name, gpu, pod_id))
            return {"id": pod_id, "name": name, "costPerHr": self.stock[gpu]}

    def pod(self, pod_id):
        gone = pod_id in self.terminated
        return None if gone else {"id": pod_id, "desiredStatus": "RUNNING", "runtime": {}}

    def terminate(self, pod_id):
        with self.lock:
            self.terminated.append(pod_id)


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="backcast-test-"))
        self.files = self.tmp / "pod_files"
        self.files.mkdir()
        (self.files / "install.sh").write_text('#!/bin/bash\ntouch "$1/INSTALLED"\n')
        (self.files / "serve.sh").write_text(FAKE_SERVE)
        (self.files / "fake_service.py").write_text(FAKE_SERVICE)
        real = Path(pod_module.__file__).parent / "pod_files" / "pod_status.py"
        (self.files / "pod_status.py").write_text(real.read_text().replace('b"--worker-index" in', 'b"fake_service.py" in'))
        self.inference = self.tmp / "inference"
        self.inference.mkdir()
        (self.inference / "marker").write_text("x")
        self.saved = pod_module.FILES, pod_module.INFERENCE, pod_module.SYSTEM_PACKAGES
        pod_module.FILES, pod_module.INFERENCE = self.files, self.inference
        self.footage = self.tmp / "footage" / "cam"
        self.footage.mkdir(parents=True)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        pod_module.FILES, pod_module.INFERENCE, pod_module.SYSTEM_PACKAGES = self.saved
        subprocess.run(["pkill", "-f", f"[f]ake_service.py {self.tmp}"], capture_output=True)
        subprocess.run(["rm", "-rf", str(self.tmp)])

    def videos(self, *contents, start=0):
        for index, content in enumerate(contents):
            (self.footage / f"v{index + start:03d}.mp4").write_text(str(content) + "\n")

    def make(self, *, pods=2, budget=100.0, stock=None, gpus=None, **overrides):
        gpus = gpus or (config_module.GpuChoice("GPU A", 1.0, 2),)
        config = config_module.Config(
            footage_roots=(self.footage,), output_dir=self.tmp / "out", run_dir=self.tmp / "run", gpus=gpus, pods=pods,
            max_total_usd=budget, ready_videos=2, uploads_per_pod=2, **overrides)
        ledger = Ledger(config.run_dir / "state.sqlite")
        api = FakeApi(stock or {"GPU A": 1.0})

        def connect(pod):
            if pod.pod_id in api.terminated:
                return None
            pod.root = str(self.tmp / "pods" / pod.pod_id)
            return LocalRemote()

        controller.scan(config, ledger)
        fleet = Controller(config, ledger, api, connect=connect, poll_seconds=0.05, retry_stock_seconds=0.1)
        return config, ledger, api, fleet

    def run_fleet(self, fleet, timeout=60):
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("code", fleet.run()), daemon=True)
        thread.start()
        thread.join(timeout)
        self.assertFalse(thread.is_alive(), "fleet did not finish")
        return result["code"]

    def test_every_video_is_annotated_once_and_verified(self):
        self.videos(*[5] * 12)
        config, ledger, api, fleet = self.make(pods=3)
        self.assertEqual(self.run_fleet(fleet), 0)
        counts = ledger.counts()
        self.assertEqual((counts[DONE][0], counts[PENDING][0], counts[FAILED][0]), (12, 0, 0))
        outputs = sorted(config.output_dir.rglob("*.jsonl.zst"))
        self.assertEqual(len(outputs), 12)
        self.assertEqual(pod_module.count_lines(outputs[0]), 5)
        self.assertTrue(outputs[0].as_posix().endswith("cam/v000.mp4.jsonl.zst"))
        lines = [json.loads(line) for line in (config.output_dir / "_manifest.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(row["relative"] for row in lines), sorted(f"cam/v{i:03d}.mp4" for i in range(12)))
        self.assertEqual({row["frames"] for row in lines}, {5})
        self.assertEqual(sorted(api.terminated), sorted(pod_id for _, _, pod_id in api.deployed))
        self.assertFalse(ledger.pods(open_only=True))
        self.assertFalse(list(config.output_dir.rglob("*.part")))

    def test_rerun_skips_finished_videos_and_picks_up_new_ones(self):
        self.videos(*[2] * 4)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.videos(2, start=4)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 5)
        self.assertEqual(len((config.output_dir / "_manifest.jsonl").read_text().splitlines()), 5)

    def test_an_undecodable_video_is_set_aside_and_the_rest_finish(self):
        self.videos(3, "corrupt", 3, 3)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 3)
        counts = ledger.counts()
        self.assertEqual((counts[DONE][0], counts[FAILED][0]), (3, 1))
        self.assertIn("v001", ledger.failures()[0]["relative"])
        self.assertEqual(ledger.retry_failed(), 1)
        self.assertEqual(ledger.counts()[PENDING][0], 1)

    def test_a_crashed_service_restarts_and_requeues_its_video(self):
        self.videos("crash", 3)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 2)

    def test_a_lost_pod_gives_its_videos_back_and_a_new_pod_takes_over(self):
        self.videos(*[2] * 6)
        saved = pod_module.UNREACHABLE_SECONDS
        pod_module.UNREACHABLE_SECONDS = 0.2
        self.addCleanup(setattr, pod_module, "UNREACHABLE_SECONDS", saved)
        config, ledger, api, fleet = self.make(pods=1)
        original = LocalRemote.run
        broken = {"first": True}

        def flaky(remote, command, timeout=120, stdin=None):
            # The first pod dies as soon as it has been given work.
            if "pod_status.py" in command and "/pods/id0" in command and broken["first"]:
                if ledger.assigned_to(ledger.pods()[0]["name"]):
                    return subprocess.CompletedProcess(command, 255, "", "connection lost")
            return original(remote, command, timeout, stdin)

        LocalRemote.run = flaky
        self.addCleanup(setattr, LocalRemote, "run", original)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertGreaterEqual(len(api.deployed), 2)
        self.assertEqual(ledger.counts()[DONE][0], 6)
        self.assertIn("id0", api.terminated)

    def test_pods_that_keep_failing_stop_the_renting(self):
        # The real pipeline once failed this way on every start: the fleet must not pay for pod after pod.
        self.videos(*[2] * 4)
        (self.files / "install.sh").write_text('#!/bin/bash\necho "pip broke" >&2\nexit 1\n')
        config, ledger, api, fleet = self.make(pods=2)
        self.assertEqual(self.run_fleet(fleet), 4)
        self.assertLessEqual(len(api.deployed), controller.MAX_FUTILE_PODS + config.pods)
        self.assertEqual(sorted(api.terminated), sorted(pod_id for _, _, pod_id in api.deployed))
        self.assertEqual(ledger.counts()[PENDING][0], 4)
        self.assertEqual(ledger.counts()[DONE][0], 0)

    def test_a_service_that_never_starts_stops_the_renting(self):
        self.videos(*[2] * 4)
        (self.files / "serve.sh").write_text('#!/bin/bash\necho 999999 > "$1/service.pid"\n')
        saved = pod_module.MAX_SERVICE_RESTARTS
        pod_module.MAX_SERVICE_RESTARTS = 1
        self.addCleanup(setattr, pod_module, "MAX_SERVICE_RESTARTS", saved)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 4)
        self.assertLessEqual(len(api.deployed), controller.MAX_FUTILE_PODS)

    def test_only_restarts_close_together_fail_a_pod(self):
        config, ledger, api, _ = self.make(pods=1)
        pod = pod_module.Pod(name="p", pod_id="x", gpu="g", processes=1, created=time.time(), config=config, ledger=ledger, api=api,
                             connect=None, sink=None, work_left=lambda: True)
        self.addCleanup(pod.uploads.shutdown)
        pod.remote = type("Remote", (), {"run": lambda self, *a, **k: subprocess.CompletedProcess(a, 0, "", "")})()
        pod.start_service = lambda: None
        # A rare fault every couple of hours, for as long as the job runs.
        for _ in range(pod_module.MAX_SERVICE_RESTARTS * 3):
            pod.restart_service("fault")
            pod.last_restart -= pod_module.SERVICE_RESTART_WINDOW_SECONDS + 1
        pod.last_restart = 0.0
        pod.restart_service("fault")
        for _ in range(pod_module.MAX_SERVICE_RESTARTS - 1):
            pod.restart_service("fault")
        with self.assertRaises(pod_module.PodFailure):
            pod.restart_service("fault")

    def test_a_rerun_after_fixing_the_cause_starts_clean(self):
        self.videos(*[2] * 2)
        install = self.files / "install.sh"
        good = install.read_text()
        install.write_text('#!/bin/bash\nexit 1\n')
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 4)
        install.write_text(good)
        time.sleep(0.01)
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 2)

    def test_a_network_blip_while_fetching_keeps_the_pod_and_the_video(self):
        self.videos(*[2] * 3)
        config, ledger, api, fleet = self.make(pods=1)
        original = LocalRemote.get_stream
        blips = {"left": 2}

        def flaky(remote, command, local, timeout=3600):
            if blips["left"]:
                blips["left"] -= 1
                raise subprocess.TimeoutExpired(command, timeout)
            return original(remote, command, local, timeout)

        LocalRemote.get_stream = flaky
        self.addCleanup(setattr, LocalRemote, "get_stream", original)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 3)
        self.assertEqual(len(api.deployed), 1)

    def test_a_pod_that_runpod_already_removed_counts_as_closed(self):
        self.videos(1)
        config, ledger, api, fleet = self.make(pods=1)
        original = api.terminate

        def refuse(pod_id):
            original(pod_id)
            raise RuntimeError("pod not found")

        api.terminate = refuse
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.pods(open_only=True), [])

    def test_one_bad_pod_among_good_ones_does_not_stop_the_fleet(self):
        self.videos(*[2] * 8)
        install = self.files / "install.sh"
        good = install.read_text()
        install.write_text('#!/bin/bash\nif [[ "$1" == */id0 ]]; then exit 1; fi\ntouch "$1/INSTALLED"\n')
        config, ledger, api, fleet = self.make(pods=1)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 8)
        self.assertEqual(len(api.deployed), 2)

    def test_the_budget_stops_the_fleet_and_keeps_the_queue(self):
        self.videos(*[1] * 3)
        config, ledger, api, fleet = self.make(pods=1, budget=0.001, stock={"GPU A": 1.0})
        ledger.add_pod("earlier", "old", "GPU A", 2, 3600.0, time.time() - 1)  # already over budget
        ledger.update_pod("earlier", ended=time.time())
        self.assertEqual(self.run_fleet(fleet), 3)
        self.assertEqual(api.deployed, [])
        self.assertEqual(ledger.counts()[PENDING][0], 3)

    def test_gpu_preference_falls_through_stock_and_price_caps_and_retries(self):
        self.videos(*[1] * 2)
        gpus = (config_module.GpuChoice("Sold out", 1.0, 2), config_module.GpuChoice("Too dear", 0.5, 2), config_module.GpuChoice("GPU A", 1.0, 2))
        config, ledger, api, fleet = self.make(pods=1, gpus=gpus, stock={"Sold out": None, "Too dear": 0.9, "GPU A": 0.8})
        api.deploy_errors = 2
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual([gpu for _, gpu, _ in api.deployed], ["GPU A"])

    def test_max_pods_limits_a_gpu_type(self):
        self.videos(*[1] * 3)
        gpus = (config_module.GpuChoice("GPU A", 1.0, 2, max_pods=1), config_module.GpuChoice("GPU B", 1.0, 2))
        config, ledger, api, fleet = self.make(pods=2, gpus=gpus, stock={"GPU A": 0.5, "GPU B": 0.6})
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertLessEqual(sum(gpu == "GPU A" for _, gpu, _ in api.deployed[:2]), 1)

    def test_pilot_annotates_only_its_evenly_spaced_sample(self):
        self.videos(*[1] * 20)
        config, ledger, api, fleet = self.make(pods=1)
        fleet.allowed = controller.load_pilot(config, ledger, 4)
        self.assertEqual(self.run_fleet(fleet), 0)
        done = [row["relative"] for row in ledger._run("SELECT relative FROM videos WHERE state='done' ORDER BY relative")]
        self.assertEqual(done, [f"cam/v{i:03d}.mp4" for i in (0, 5, 10, 15)])
        self.assertEqual(ledger.counts()[PENDING][0], 16)

    def test_a_restarted_controller_adopts_a_live_pod(self):
        self.videos(*[1] * 3)
        config, ledger, api, fleet = self.make(pods=1)
        name, pod_id = "backcast-old", "id-old"
        api.deployed.append((name, "GPU A", pod_id))
        ledger.add_pod(name, pod_id, "GPU A", 2, 1.0, time.time())
        root = self.tmp / "pods" / pod_id
        for folder in ("code", "queue/ready", "queue/incoming", "out", "logs"):
            (root / folder).mkdir(parents=True)
        subprocess.run(["cp", "-r", str(self.files), str(root / "code" / "pod_files")], check=True)
        (root / "INSTALLED").write_text("")
        # The old controller had assigned this video, but it never reached the pod.
        lost = ledger.claim(name)
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertEqual(ledger.counts()[DONE][0], 3)
        self.assertEqual(ledger.get(lost["id"])["attempts"], 2)
        self.assertEqual(api.deployed[0][0], name)
        self.assertIn(pod_id, api.terminated)

    def test_a_surviving_pod_of_a_family_no_longer_allowed_is_ended(self):
        self.videos(*[1] * 3)
        allowed = config_module.GpuChoice("GPU B", 1.0, 2, family="ada")
        config, ledger, api, fleet = self.make(pods=1, gpus=(allowed,), stock={"GPU A": 0.5, "GPU B": 0.5})
        ledger.add_pod("old", "id-old", "GPU A", 2, 1.0, time.time())
        held = ledger.claim("old")
        self.assertEqual(self.run_fleet(fleet), 0)
        self.assertIn("id-old", api.terminated)
        self.assertEqual({gpu for _, gpu, _ in api.deployed}, {"GPU B"})
        self.assertEqual(ledger.counts()[DONE][0], 3)
        self.assertEqual(ledger.get(held["id"])["state"], DONE)

    def test_terminate_closes_every_open_pod(self):
        self.videos(1)
        config, ledger, api, fleet = self.make()
        ledger.add_pod("a", "id-a", "GPU A", 2, 1.0, time.time())
        row = ledger.claim("a")
        self.assertEqual(controller.terminate_open_pods(config, ledger, api), 1)
        self.assertEqual(api.terminated, ["id-a"])
        self.assertEqual(ledger.pods(open_only=True), [])
        self.assertEqual(ledger.get(row["id"])["state"], PENDING)


class GpuFamilyTest(unittest.TestCase):
    def load(self, setting):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fleet.toml"
            path.write_text('[footage]\nroots = ["x"]\n[output]\ndir = "y"\n[fleet]\npods = 1\nmax_total_usd = 5\n' + setting +
                            '[[fleet.gpu]]\nid = "A"\nfamily = "Ada"\nmax_usd_per_hour = 1\nprocesses = 8\n'
                            '[[fleet.gpu]]\nid = "B"\nfamily = "ampere"\nmax_usd_per_hour = 1\nprocesses = 8\n'
                            '[[fleet.gpu]]\nid = "C"\nmax_usd_per_hour = 1\nprocesses = 8\n')
            return config_module.load(path)

    def test_mixing_is_the_default(self):
        config = self.load("")
        self.assertEqual(config.gpu_family, "any")
        self.assertEqual([g.id for g in config.gpus], ["A", "B", "C"])

    def test_one_family_keeps_only_its_gpus_in_order(self):
        self.assertEqual([g.id for g in self.load('gpu_family = "ampere"\n').gpus], ["B"])
        self.assertEqual([g.id for g in self.load('gpu_family = "ADA"\n').gpus], ["A"])

    def test_an_unknown_family_is_refused_up_front(self):
        with self.assertRaisesRegex(ValueError, "ada.*ampere"):
            self.load('gpu_family = "blackwell"\n')

    def test_every_gpu_in_the_example_names_a_family(self):
        example = tomllib.loads((Path(__file__).resolve().parent.parent / "fleet.example.toml").read_text())
        self.assertEqual(example["fleet"]["gpu_family"], "any")
        for gpu in example["fleet"]["gpu"]:
            self.assertIn(gpu["family"], ("ada", "ampere", "blackwell"), gpu["id"])


class ConfigTest(unittest.TestCase):
    def test_example_config_loads(self):
        config = config_module.load(Path(__file__).resolve().parent.parent / "fleet.example.toml")
        self.assertTrue(config.gpus)
        self.assertEqual(config.cloud, "SECURE")
        self.assertGreater(config.max_total_usd, 0)

    def test_bad_config_names_the_problem(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fleet.toml"
            path.write_text('[footage]\nroots = []\n[output]\ndir = "x"\n[fleet]\npods = 1\nmax_total_usd = 5\n[[fleet.gpu]]\nid = "x"\nmax_usd_per_hour = 1\nprocesses = 2\n')
            with self.assertRaisesRegex(ValueError, "roots"):
                config_module.load(path)


class FootageTest(unittest.TestCase):
    def test_discover_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cam"
            (root / "day2").mkdir(parents=True)
            for name in ("b.MP4", "a.mkv", ".hidden.mp4", "notes.txt", "day2/c.avi", "skip_me.mp4"):
                (root / name).write_text("x")
            rows = footage.discover((root,), (".mp4", ".mkv", ".avi"), ("*skip_me*",))
            self.assertEqual([row[1] for row in rows], ["cam/a.mkv", "cam/b.MP4", "cam/day2/c.avi"])

    def test_remote_names_are_safe_and_recover_the_id(self):
        name = pod_module.remote_name(42, "cam 1/day (3)/video;rm -rf.mp4")
        self.assertRegex(name, r"^[A-Za-z0-9_.-]+$")
        self.assertEqual(pod_module.video_id_of(name + ".mp4"), 42)


if __name__ == "__main__":
    unittest.main()
