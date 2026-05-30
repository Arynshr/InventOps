"""
metrics/logger.py — Thin SQLite-backed metric logger for InventOps.

Usage:
    from metrics import get_logger

    logger = get_logger()                     # writes to metrics/inventops.db
    logger = get_logger(db_path=":memory:")   # in-memory (tests)
    logger = get_logger(noop=True)            # silent no-op (disable entirely)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    seed            INTEGER NOT NULL,
    agent           TEXT    NOT NULL DEFAULT 'llm',
    score           REAL    NOT NULL,
    success         INTEGER NOT NULL,   -- 0/1
    steps           INTEGER NOT NULL,
    ts              REAL    NOT NULL    -- unix timestamp
);

CREATE TABLE IF NOT EXISTS steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    seed            INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    action_type     TEXT    NOT NULL,
    reward          REAL    NOT NULL,
    fulfillment     REAL,
    holding_cost    REAL,
    stockout_pen    REAL,
    order_cost      REAL,
    transfer_cost   REAL,
    capacity_pen    REAL,
    bullwhip_pen    REAL,
    failure_reason  TEXT,
    llm_latency_ms  REAL,
    ts              REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS rlvr_rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    round_num       INTEGER NOT NULL,
    mean_score      REAL    NOT NULL,
    min_score       REAL    NOT NULL,
    max_score       REAL    NOT NULL,
    failure_summary TEXT,               -- JSON list
    ts              REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_task  ON episodes(task_id);
CREATE INDEX IF NOT EXISTS idx_steps_run      ON steps(run_id, task_id, seed);
CREATE INDEX IF NOT EXISTS idx_rlvr_run       ON rlvr_rounds(run_id, task_id);
"""


class MetricLogger:
    """
    Thread-safe SQLite metric logger.

    All public methods are safe to call from multiple threads.
    Pass noop=True to create a silent no-op instance (useful in tests).
    """

    def __init__(self, db_path: str = "metrics/inventops.db", noop: bool = False):
        self._noop = noop
        self._db_path = db_path  # stored so get_logger() can detect conflicting re-init
        if noop:
            return

        path = Path(db_path)
        if db_path != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False + our own lock → safe for multi-threaded use
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── internal ──────────────────────────────────────────────────────────────

    def _exec(self, sql: str, params: tuple) -> None:
        if self._noop:
            return
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ── public API ────────────────────────────────────────────────────────────

    def log_step(
        self,
        *,
        run_id: str,
        task_id: str,
        seed: int,
        step: int,
        action_type: str,
        reward: float,
        failure_reason: Optional[str] = None,
        llm_latency_ms: Optional[float] = None,
        # reward breakdown (from StepReward if available)
        fulfillment: Optional[float] = None,
        holding_cost: Optional[float] = None,
        stockout_pen: Optional[float] = None,
        order_cost: Optional[float] = None,
        transfer_cost: Optional[float] = None,
        capacity_pen: Optional[float] = None,
        bullwhip_pen: Optional[float] = None,
    ) -> None:
        self._exec(
            """INSERT INTO steps
               (run_id, task_id, seed, step, action_type, reward,
                fulfillment, holding_cost, stockout_pen, order_cost,
                transfer_cost, capacity_pen, bullwhip_pen,
                failure_reason, llm_latency_ms, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, task_id, seed, step, action_type, reward,
                fulfillment, holding_cost, stockout_pen, order_cost,
                transfer_cost, capacity_pen, bullwhip_pen,
                failure_reason, llm_latency_ms, time.time(),
            ),
        )

    def log_episode(
        self,
        *,
        run_id: str,
        task_id: str,
        seed: int,
        score: float,
        success: bool,
        steps: int,
        agent: str = "llm",
    ) -> None:
        self._exec(
            """INSERT INTO episodes
               (run_id, task_id, seed, agent, score, success, steps, ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, task_id, seed, agent, score, int(success), steps, time.time()),
        )

    def log_rlvr_round(
        self,
        *,
        run_id: str,
        task_id: str,
        round_num: int,
        mean_score: float,
        min_score: float,
        max_score: float,
        failure_reasons: Optional[list[str]] = None,
    ) -> None:
        import json
        summary = json.dumps(list(set(failure_reasons or [])))
        self._exec(
            """INSERT INTO rlvr_rounds
               (run_id, task_id, round_num, mean_score, min_score, max_score,
                failure_summary, ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, task_id, round_num, mean_score, min_score, max_score, summary, time.time()),
        )

    def close(self) -> None:
        if not self._noop:
            with self._lock:
                self._conn.close()


# ── Singleton helper ─────────────────────────────────────────────────────────

_default_logger: Optional[MetricLogger] = None
_logger_lock = threading.Lock()


def get_logger(
    db_path: str = "metrics/inventops.db",
    noop: bool = False,
) -> MetricLogger:
    """Return (or create) the process-wide default MetricLogger.

    The first call initialises the singleton with the given ``db_path`` and
    ``noop`` values.  Subsequent calls with *different* arguments are ignored
    (the original singleton is returned) and a warning is emitted so that
    accidental misconfiguration (e.g. tests expecting ``:memory:``) surfaces
    immediately instead of silently writing to the wrong database.
    """
    global _default_logger
    with _logger_lock:
        if _default_logger is None:
            _default_logger = MetricLogger(db_path=db_path, noop=noop)
        else:
            existing_path = getattr(_default_logger, "_db_path", None)
            existing_noop = getattr(_default_logger, "_noop", None)
            if db_path != existing_path or noop != existing_noop:
                _log.warning(
                    "get_logger() called with db_path=%r noop=%r but singleton already "
                    "initialised with db_path=%r noop=%r — returning existing logger.",
                    db_path, noop, existing_path, existing_noop,
                )
    return _default_logger
