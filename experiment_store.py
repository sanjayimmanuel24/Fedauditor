"""Persistence for experiment runs.

Previously the simulation lived in a module-level global, so every logged-in
clinician shared one state and nothing survived a restart. Rounds, calibration
comparisons, fairness audits and valuations are now written to SQLite, which
makes results reproducible and exportable.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from typing import Dict, List

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_json TEXT,
    data_source TEXT,
    partition_mode TEXT
);

CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    metrics_json TEXT,
    local_json TEXT,
    cm_json TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS fairness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS valuations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_rounds_run ON rounds(run_id);
CREATE INDEX IF NOT EXISTS idx_calibration_run ON calibration(run_id);
CREATE INDEX IF NOT EXISTS idx_fairness_run ON fairness(run_id);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_store() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def create_run(
    name: str,
    config: dict | None = None,
    data_source: str = "pima",
    partition_mode: str = "age",
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO runs (name, config_json, data_source, partition_mode)"
            " VALUES (?, ?, ?, ?)",
            (name, json.dumps(config or {}), data_source, partition_mode),
        )
        return int(cursor.lastrowid)


def append_round(run_id: int, round_no: int, metrics: dict, local: list, cm: list) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO rounds (run_id, round_no, metrics_json, local_json, cm_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (run_id, round_no, json.dumps(metrics), json.dumps(local), json.dumps(cm)),
        )


def _save_payload(table: str, run_id: int, round_no: int, payload: dict) -> None:
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO {table} (run_id, round_no, payload_json) VALUES (?, ?, ?)",
            (run_id, round_no, json.dumps(payload)),
        )


def save_calibration(run_id: int, round_no: int, payload: dict) -> None:
    _save_payload("calibration", run_id, round_no, payload)


def save_fairness(run_id: int, round_no: int, payload: dict) -> None:
    _save_payload("fairness", run_id, round_no, payload)


def save_valuation(run_id: int, round_no: int, payload: dict) -> None:
    _save_payload("valuations", run_id, round_no, payload)


def save_explanation(run_id: int, round_no: int, payload: dict) -> None:
    _save_payload("explanations", run_id, round_no, payload)


def get_rounds(run_id: int) -> List[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT round_no, metrics_json, local_json, cm_json FROM rounds"
            " WHERE run_id = ? ORDER BY round_no",
            (run_id,),
        ).fetchall()
    return [
        {
            "round": row["round_no"],
            "metrics": json.loads(row["metrics_json"]),
            "local_accuracies": json.loads(row["local_json"]),
            "confusion_matrix": json.loads(row["cm_json"]),
        }
        for row in rows
    ]


def latest_payload(table: str, run_id: int) -> dict | None:
    if table not in {"calibration", "fairness", "valuations", "explanations"}:
        raise ValueError(f"unknown table {table}")
    with _connect() as conn:
        row = conn.execute(
            f"SELECT payload_json FROM {table} WHERE run_id = ?"
            " ORDER BY round_no DESC, id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def list_runs(limit: int = 20) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, data_source, partition_mode FROM runs"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def rounds_csv(run_id: int) -> str:
    """Audit-friendly export of the per-round metric history."""
    rounds = get_rounds(run_id)
    buffer = io.StringIO()
    fieldnames = [
        "round",
        "accuracy",
        "auc",
        "brier",
        "nll",
        "ece",
        "mce",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for entry in rounds:
        metrics = entry["metrics"]
        writer.writerow({key: metrics.get(key) for key in fieldnames})
    return buffer.getvalue()


def run_summary(run_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, config_json, data_source, partition_mode"
            " FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    entry = dict(row)
    entry["config"] = json.loads(entry.pop("config_json") or "{}")
    return entry
