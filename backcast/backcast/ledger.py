"""The durable work list: every video is annotated once, and a restart resumes where the last run stopped."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import threading
import time

PENDING, ASSIGNED, DONE, FAILED = "pending", "assigned", "done", "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, relative TEXT NOT NULL, bytes INTEGER NOT NULL,
  frames INTEGER, width INTEGER, height INTEGER, fps REAL, codec TEXT, probed INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending', pod TEXT, attempts INTEGER NOT NULL DEFAULT 0,
  frames_done INTEGER, gpu TEXT, error TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS videos_state ON videos(state);
CREATE TABLE IF NOT EXISTS pods(
  name TEXT PRIMARY KEY, pod_id TEXT, gpu TEXT, processes INTEGER, rate REAL, created REAL, ended REAL,
  state TEXT NOT NULL, detail TEXT, frames_done INTEGER NOT NULL DEFAULT 0, first_frame REAL, last_frame REAL);
"""


class Ledger:
    def __init__(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _run(self, sql: str, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # ------------------------------------------------------------- videos
    def add_videos(self, rows) -> int:
        """Register new files; known paths keep their state. Returns how many were new."""
        added = 0
        with self.lock:
            for path, relative, size in rows:
                added += self.db.execute("INSERT OR IGNORE INTO videos(path, relative, bytes, updated) VALUES(?,?,?,?)",
                                         (path, relative, size, time.time())).rowcount
        return added

    def unprobed(self, limit=None) -> list[str]:
        sql = "SELECT path FROM videos WHERE probed=0 ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
        return [row["path"] for row in self._run(sql)]

    def set_probe(self, path: str, info: dict) -> None:
        self._run("UPDATE videos SET probed=1, frames=?, width=?, height=?, fps=?, codec=? WHERE path=?",
                  (info.get("frames"), info.get("width"), info.get("height"), info.get("fps"), info.get("codec"), path))

    def limit_to(self, count: int) -> list[int]:
        """Evenly spaced video ids across the sorted archive: a representative pilot."""
        rows = self._run("SELECT id FROM videos WHERE state=? ORDER BY relative", (PENDING,))
        if not rows:
            return []
        step = max(1, len(rows) // count)
        return [row["id"] for row in rows[::step][:count]]

    def claim(self, pod: str, allowed: set[int] | None = None) -> sqlite3.Row | None:
        """Assign the next pending video to a pod, or None when none is left."""
        with self.lock:
            rows = self.db.execute("SELECT * FROM videos WHERE state=? ORDER BY relative", (PENDING,))
            for row in rows:
                if allowed is None or row["id"] in allowed:
                    self.db.execute("UPDATE videos SET state=?, pod=?, attempts=attempts+1, updated=? WHERE id=?",
                                    (ASSIGNED, pod, time.time(), row["id"]))
                    return row
        return None

    def pending_count(self, allowed: set[int] | None = None) -> int:
        rows = self._run("SELECT id FROM videos WHERE state=?", (PENDING,))
        return sum(allowed is None or row["id"] in allowed for row in rows)

    def get(self, video_id: int) -> sqlite3.Row | None:
        rows = self._run("SELECT * FROM videos WHERE id=?", (video_id,))
        return rows[0] if rows else None

    def finish(self, video_id: int, frames: int, gpu: str) -> None:
        self._run("UPDATE videos SET state=?, frames_done=?, gpu=?, error=NULL, updated=? WHERE id=?",
                  (DONE, frames, gpu, time.time(), video_id))

    def requeue(self, video_id: int, error: str, max_attempts: int) -> str:
        """Back to pending for another try, or failed once its attempts run out. Returns the new state."""
        with self.lock:
            row = self.db.execute("SELECT attempts FROM videos WHERE id=?", (video_id,)).fetchone()
            state = FAILED if row["attempts"] >= max_attempts else PENDING
            self.db.execute("UPDATE videos SET state=?, pod=NULL, error=?, updated=? WHERE id=?", (state, error, time.time(), video_id))
        return state

    def fail(self, video_id: int, error: str) -> None:
        self._run("UPDATE videos SET state=?, pod=NULL, error=?, updated=? WHERE id=?", (FAILED, error, time.time(), video_id))

    def assigned_to(self, pod: str) -> list[sqlite3.Row]:
        return self._run("SELECT * FROM videos WHERE state=? AND pod=?", (ASSIGNED, pod))

    def release_pod(self, pod: str, error: str, max_attempts: int) -> int:
        rows = self.assigned_to(pod)
        for row in rows:
            self.requeue(row["id"], error, max_attempts)
        return len(rows)

    def retry_failed(self) -> int:
        with self.lock:
            return self.db.execute("UPDATE videos SET state=?, attempts=0, error=NULL WHERE state=?", (PENDING, FAILED)).rowcount

    def counts(self) -> dict:
        totals = {PENDING: [0, 0, 0], ASSIGNED: [0, 0, 0], DONE: [0, 0, 0], FAILED: [0, 0, 0]}
        for row in self._run("SELECT state, COUNT(*) n, COALESCE(SUM(frames),0) f, COALESCE(SUM(bytes),0) b FROM videos GROUP BY state"):
            totals[row["state"]] = [row["n"], row["f"], row["b"]]
        return totals

    def failures(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._run("SELECT relative, error, attempts FROM videos WHERE state=? ORDER BY updated DESC LIMIT ?", (FAILED, limit))

    # --------------------------------------------------------------- pods
    def add_pod(self, name, pod_id, gpu, processes, rate, created) -> None:
        self._run("INSERT INTO pods(name, pod_id, gpu, processes, rate, created, state) VALUES(?,?,?,?,?,?,?)",
                  (name, pod_id, gpu, processes, rate, created, "starting"))

    def update_pod(self, name, **fields) -> None:
        columns = ", ".join(f"{key}=?" for key in fields)
        self._run(f"UPDATE pods SET {columns} WHERE name=?", (*fields.values(), name))

    def pods(self, *, open_only=False) -> list[sqlite3.Row]:
        return self._run("SELECT * FROM pods" + (" WHERE ended IS NULL" if open_only else "") + " ORDER BY created")

    def spent(self, now: float | None = None) -> float:
        """Estimated dollars so far: each pod's hourly rate over its lifetime."""
        now = time.time() if now is None else now
        total = 0.0
        for pod in self.pods():
            if pod["rate"] and pod["created"]:
                total += pod["rate"] * ((pod["ended"] or now) - pod["created"]) / 3600
        return total
