"""
dashboard/queries.py — SQL query helpers for the InventOps Streamlit dashboard.
All functions return pandas DataFrames.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = "metrics/inventops.db"


def _conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


# ── Overview ──────────────────────────────────────────────────────────────────

def summary_kpis(db_path: str = DEFAULT_DB) -> dict:
    """Return top-level KPI numbers for the Overview tab."""
    con = _conn(db_path)
    try:
        ep = pd.read_sql("SELECT * FROM episodes", con)
        st = pd.read_sql("SELECT * FROM steps", con)
        rl = pd.read_sql("SELECT * FROM rlvr_rounds", con)
    finally:
        con.close()

    if ep.empty:
        return {
            "total_episodes": 0, "success_rate": 0.0,
            "mean_score": 0.0, "total_steps": 0,
            "rlvr_rounds": 0, "parse_errors": 0,
        }

    return {
        "total_episodes": len(ep),
        "success_rate":   round(ep["success"].mean() * 100, 1),
        "mean_score":     round(ep["score"].mean(), 4),
        "total_steps":    len(st),
        "rlvr_rounds":    len(rl),
        "parse_errors":   int(st["failure_reason"].notna().sum()),
    }


def score_by_task(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        return pd.read_sql(
            """SELECT task_id, agent,
                      AVG(score)  AS mean_score,
                      MIN(score)  AS min_score,
                      MAX(score)  AS max_score,
                      COUNT(*)    AS n_episodes
               FROM episodes
               GROUP BY task_id, agent
               ORDER BY task_id, agent""",
            con,
        )
    finally:
        con.close()


def recent_runs(db_path: str = DEFAULT_DB, limit: int = 50) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        return pd.read_sql(
            f"""SELECT run_id, task_id, agent, seed, score, success, steps,
                       datetime(ts, 'unixepoch', 'localtime') AS recorded_at
                FROM episodes
                ORDER BY ts DESC
                LIMIT {limit}""",
            con,
        )
    finally:
        con.close()


# ── Episodes tab ──────────────────────────────────────────────────────────────

def reward_curve(
    db_path: str = DEFAULT_DB,
    task_id: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Return per-step rewards for a specific (task, seed, run_id) combo."""
    wheres, params = [], []
    if task_id:
        wheres.append("task_id = ?"); params.append(task_id)
    if seed is not None:
        wheres.append("seed = ?"); params.append(seed)
    if run_id:
        wheres.append("run_id = ?"); params.append(run_id)

    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    con = _conn(db_path)
    try:
        return pd.read_sql(
            f"SELECT * FROM steps {where_clause} ORDER BY step",
            con, params=params,
        )
    finally:
        con.close()


def action_distribution(
    db_path: str = DEFAULT_DB,
    task_id: str | None = None,
) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        if task_id:
            return pd.read_sql(
                "SELECT action_type, COUNT(*) AS count FROM steps WHERE task_id = ? GROUP BY action_type",
                con, params=[task_id],
            )
        return pd.read_sql(
            "SELECT action_type, COUNT(*) AS count FROM steps GROUP BY action_type",
            con,
        )
    finally:
        con.close()


def failure_reasons(db_path: str = DEFAULT_DB, task_id: str | None = None) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        if task_id:
            return pd.read_sql(
                """SELECT failure_reason, COUNT(*) AS occurrences
                    FROM steps
                    WHERE failure_reason IS NOT NULL AND task_id = ?
                    GROUP BY failure_reason
                    ORDER BY occurrences DESC""",
                con, params=[task_id],
            )
        return pd.read_sql(
            """SELECT failure_reason, COUNT(*) AS occurrences
                FROM steps
                WHERE failure_reason IS NOT NULL
                GROUP BY failure_reason
                ORDER BY occurrences DESC""",
            con,
        )
    finally:
        con.close()


def reward_breakdown_agg(db_path: str = DEFAULT_DB, task_id: str | None = None) -> pd.DataFrame:
    """Average reward components across all steps."""
    _AGG_SQL = """SELECT AVG(fulfillment)   AS fulfillment,
                         AVG(holding_cost)  AS holding_cost,
                         AVG(stockout_pen)  AS stockout_penalty,
                         AVG(order_cost)    AS order_cost,
                         AVG(transfer_cost) AS transfer_cost,
                         AVG(capacity_pen)  AS capacity_breach,
                         AVG(bullwhip_pen)  AS bullwhip
                  FROM steps"""
    con = _conn(db_path)
    try:
        if task_id:
            df = pd.read_sql(_AGG_SQL + " WHERE task_id = ?", con, params=[task_id])
        else:
            df = pd.read_sql(_AGG_SQL, con)
    finally:
        con.close()
    return df


def available_run_ids(db_path: str = DEFAULT_DB) -> list[str]:
    con = _conn(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT run_id FROM steps ORDER BY rowid DESC LIMIT 30"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def available_seeds(db_path: str = DEFAULT_DB, task_id: str | None = None, run_id: str | None = None) -> list[int]:
    wheres, params = [], []
    if task_id:
        wheres.append("task_id = ?"); params.append(task_id)
    if run_id:
        wheres.append("run_id = ?"); params.append(run_id)
    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    con = _conn(db_path)
    try:
        rows = con.execute(
            f"SELECT DISTINCT seed FROM steps {where_clause} ORDER BY seed",
            params,
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


# ── RLVR tab ──────────────────────────────────────────────────────────────────

def rlvr_progression(db_path: str = DEFAULT_DB, task_id: str | None = None) -> pd.DataFrame:
    _RLVR_SQL = """SELECT run_id, task_id, round_num,
                          mean_score, min_score, max_score, failure_summary,
                          datetime(ts, 'unixepoch', 'localtime') AS recorded_at
                   FROM rlvr_rounds"""
    con = _conn(db_path)
    try:
        if task_id:
            return pd.read_sql(_RLVR_SQL + " WHERE task_id = ? ORDER BY ts, round_num", con, params=[task_id])
        return pd.read_sql(_RLVR_SQL + " ORDER BY ts, round_num", con)
    finally:
        con.close()


def rlvr_failure_summary(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """Flatten failure_summary JSON lists and count per failure type."""
    import json as _json
    con = _conn(db_path)
    try:
        rows = con.execute(
            "SELECT failure_summary FROM rlvr_rounds WHERE failure_summary IS NOT NULL"
        ).fetchall()
    finally:
        con.close()

    counts: dict[str, int] = {}
    for (raw,) in rows:
        try:
            items = _json.loads(raw)
            for item in items:
                counts[item] = counts.get(item, 0) + 1
        except Exception:
            pass
    if not counts:
        return pd.DataFrame(columns=["failure_reason", "count"])
    return (
        pd.DataFrame(list(counts.items()), columns=["failure_reason", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# ── Inference tab ─────────────────────────────────────────────────────────────

def llm_latency_hist(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        return pd.read_sql(
            "SELECT llm_latency_ms FROM steps WHERE llm_latency_ms IS NOT NULL",
            con,
        )
    finally:
        con.close()


def parse_error_rate_over_time(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    con = _conn(db_path)
    try:
        return pd.read_sql(
            """SELECT
                   strftime('%Y-%m-%dT%H:%M', datetime(ts, 'unixepoch', 'localtime')) AS minute,
                   COUNT(*) AS total_steps,
                   SUM(CASE WHEN failure_reason IS NOT NULL THEN 1 ELSE 0 END) AS failures
               FROM steps
               GROUP BY minute
               ORDER BY minute""",
            con,
        )
    finally:
        con.close()
