"""SshRemote against a throwaway sshd on localhost: real ssh, rsync and zstd, no cloud."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backcast.pod import count_lines  # noqa: E402
from backcast.remote import SshRemote  # noqa: E402

SSHD = shutil.which("sshd") or ("/usr/sbin/sshd" if Path("/usr/sbin/sshd").exists() else None)


@unittest.skipUnless(SSHD and shutil.which("rsync") and shutil.which("zstd"), "needs sshd, rsync and zstd")
class SshRemoteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="backcast-ssh-"))
        host_key, user_key = cls.tmp / "host_key", cls.tmp / "user_key"
        for key in (host_key, user_key):
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
        authorized = cls.tmp / "authorized_keys"
        shutil.copy(user_key.with_suffix(".pub"), authorized)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        config = cls.tmp / "sshd_config"
        config.write_text(f"Port {cls.port}\nListenAddress 127.0.0.1\nHostKey {host_key}\nAuthorizedKeysFile {authorized}\n"
                          f"PasswordAuthentication no\nStrictModes no\nUsePAM no\nPidFile {cls.tmp}/sshd.pid\n")
        cls.server = subprocess.Popen([SSHD, "-D", "-e", "-f", str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.control = cls.tmp / "cm"
        cls.control.mkdir(mode=0o700)
        cls.remote = SshRemote("127.0.0.1", cls.port, user_key, cls.tmp / "known_hosts", cls.control, user=getpass.getuser())
        for _ in range(50):
            try:
                if cls.remote.run("true", timeout=5).returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.1)
        cls.tearDownClass()
        raise unittest.SkipTest("the throwaway sshd did not accept a login here")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        subprocess.run(["ssh", "-O", "exit", "-o", f"ControlPath={cls.control}/%C", "-p", str(cls.port), "127.0.0.1"], capture_output=True)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_commands_report_output_and_exit_codes(self):
        self.assertEqual(self.remote.run("echo hello").stdout.strip(), "hello")
        self.assertEqual(self.remote.run("exit 7").returncode, 7)
        self.assertEqual(self.remote.run("cat", stdin="piped").stdout, "piped")

    def test_put_copies_a_file_and_a_directory_tree(self):
        source = self.tmp / "video.mp4"
        source.write_bytes(os.urandom(200_000))
        target = self.tmp / "remote_incoming"
        target.mkdir()
        sent = self.remote.put(source, str(target / "video.mp4.part"), bwlimit_kbps=100_000)
        self.assertEqual(sent.returncode, 0, sent.stderr)
        self.assertEqual((target / "video.mp4.part").read_bytes(), source.read_bytes())
        tree = self.tmp / "code"
        (tree / "sub").mkdir(parents=True)
        (tree / "sub" / "a.txt").write_text("a")
        (tree / "__pycache__").mkdir()
        (tree / "__pycache__" / "skip.pyc").write_text("x")
        sent = self.remote.put(f"{tree}/", f"{self.tmp}/remote_code/", recursive=True)
        self.assertEqual(sent.returncode, 0, sent.stderr)
        self.assertEqual((self.tmp / "remote_code" / "sub" / "a.txt").read_text(), "a")
        self.assertFalse((self.tmp / "remote_code" / "__pycache__").exists())

    def test_fetching_a_compressed_annotation_file_keeps_every_record(self):
        records = self.tmp / "video.jsonl"
        records.write_text("".join('{"frame": %d}\n' % i for i in range(5000)))
        local = self.tmp / "fetched.zst"
        result = self.remote.get_stream(f"zstd -3 -q -c {records}", local)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(count_lines(local), 5000)
        self.assertLess(local.stat().st_size, records.stat().st_size / 3)
        self.assertNotEqual(self.remote.get_stream("zstd -3 -q -c /no/such/file", self.tmp / "bad.zst").returncode, 0)

    def test_a_damaged_download_is_detected(self):
        local = self.tmp / "damaged.zst"
        records = self.tmp / "ok.jsonl"
        records.write_text("x\n" * 1000)
        self.remote.get_stream(f"zstd -3 -q -c {records}", local)
        data = local.read_bytes()
        local.write_bytes(data[: len(data) // 2])
        self.assertEqual(count_lines(local), -1)


if __name__ == "__main__":
    unittest.main()
