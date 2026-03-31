"""Database operations for the multi-layer autoresearch system."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from lib.paths import db_path

DB_PATH = db_path()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    conn = get_connection()
    conn.executescript(
        """
        -- Skill versions track the evolution of each layer's skill document.
        CREATE TABLE IF NOT EXISTS skill_versions (
            id INTEGER PRIMARY KEY,
            layer TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            version TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            parent_version_id INTEGER,
            created_by_run_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (parent_version_id) REFERENCES skill_versions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_skill_versions_layer_skill
        ON skill_versions(layer, skill_name, created_at);

        -- Experiments track every experiment across all three layers.
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            run_id TEXT UNIQUE NOT NULL,
            layer TEXT NOT NULL CHECK (layer IN ('feature', 'architecture', 'training')),

            -- What was being tested
            description TEXT,
            parent_run_id TEXT,

            -- Skill version used for this experiment
            skill_version_id INTEGER,

            -- Results
            accuracy REAL,
            log_loss REAL,
            sharpe REAL,
            profit_weighted_accuracy REAL,
            best_val_accuracy REAL,

            -- Status
            status TEXT NOT NULL CHECK (status IN ('success', 'failed', 'crash', 'keep', 'discard')),
            error_message TEXT,

            -- Artifacts
            trace_path TEXT,
            config_snapshot TEXT,

            -- Cost tracking
            duration_ms INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            model_id TEXT,

            created_at TEXT NOT NULL,
            FOREIGN KEY (skill_version_id) REFERENCES skill_versions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_experiments_layer
        ON experiments(layer, created_at);

        CREATE INDEX IF NOT EXISTS idx_experiments_status
        ON experiments(layer, status);

        -- Skill refinements link postmortems to skill updates.
        CREATE TABLE IF NOT EXISTS skill_refinements (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            old_skill_version_id INTEGER NOT NULL,
            new_skill_version_id INTEGER NOT NULL,
            refinement_summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES experiments(run_id),
            FOREIGN KEY (old_skill_version_id) REFERENCES skill_versions(id),
            FOREIGN KEY (new_skill_version_id) REFERENCES skill_versions(id)
        );

        -- Postmortems record analysis results.
        CREATE TABLE IF NOT EXISTS postmortems (
            id INTEGER PRIMARY KEY,
            layer TEXT NOT NULL,
            experiment_count INTEGER NOT NULL,
            analysis TEXT,
            changes_made TEXT,
            trace_path TEXT,
            created_at TEXT NOT NULL
        );

        -- Results log: simple append-only log similar to results.tsv in autoresearch.
        CREATE TABLE IF NOT EXISTS results_log (
            id INTEGER PRIMARY KEY,
            layer TEXT NOT NULL,
            run_id TEXT NOT NULL,
            metric_value REAL,
            metric_name TEXT NOT NULL DEFAULT 'accuracy',
            status TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def insert_skill_version(
    layer: str,
    skill_name: str,
    version: str,
    content: str,
    content_hash: str,
    parent_version_id: Optional[int] = None,
    created_by_run_id: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO skill_versions
           (layer, skill_name, version, content, content_hash,
            parent_version_id, created_by_run_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (layer, skill_name, version, content, content_hash,
         parent_version_id, created_by_run_id, _now_iso()),
    )
    conn.commit()
    version_id = int(cursor.lastrowid)
    conn.close()
    return version_id


def insert_experiment(
    *,
    run_id: str,
    layer: str,
    description: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    skill_version_id: Optional[int] = None,
    accuracy: Optional[float] = None,
    log_loss: Optional[float] = None,
    sharpe: Optional[float] = None,
    profit_weighted_accuracy: Optional[float] = None,
    best_val_accuracy: Optional[float] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    trace_path: Optional[str] = None,
    config_snapshot: Optional[str] = None,
    duration_ms: Optional[int] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost_usd: Optional[float] = None,
    model_id: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO experiments
           (run_id, layer, description, parent_run_id,
            skill_version_id,
            accuracy, log_loss, sharpe, profit_weighted_accuracy, best_val_accuracy,
            status, error_message,
            trace_path, config_snapshot,
            duration_ms, tokens_in, tokens_out, cost_usd, model_id,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, layer, description, parent_run_id,
         skill_version_id,
         accuracy, log_loss, sharpe, profit_weighted_accuracy, best_val_accuracy,
         status, error_message,
         trace_path, config_snapshot,
         duration_ms, tokens_in, tokens_out, cost_usd, model_id,
         _now_iso()),
    )
    conn.commit()
    exp_id = int(cursor.lastrowid)
    conn.close()
    return exp_id


def update_experiment_metrics(
    run_id: str,
    accuracy: Optional[float] = None,
    log_loss: Optional[float] = None,
    sharpe: Optional[float] = None,
    profit_weighted_accuracy: Optional[float] = None,
    best_val_accuracy: Optional[float] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    conn = get_connection()
    sets = []
    params = []
    if accuracy is not None:
        sets.append("accuracy = ?")
        params.append(accuracy)
    if log_loss is not None:
        sets.append("log_loss = ?")
        params.append(log_loss)
    if sharpe is not None:
        sets.append("sharpe = ?")
        params.append(sharpe)
    if profit_weighted_accuracy is not None:
        sets.append("profit_weighted_accuracy = ?")
        params.append(profit_weighted_accuracy)
    if best_val_accuracy is not None:
        sets.append("best_val_accuracy = ?")
        params.append(best_val_accuracy)
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if error_message is not None:
        sets.append("error_message = ?")
        params.append(error_message)

    if sets:
        params.append(run_id)
        conn.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE run_id = ?", params)
        conn.commit()
    conn.close()


def insert_skill_refinement(
    run_id: str,
    layer: str,
    old_skill_version_id: int,
    new_skill_version_id: int,
    refinement_summary: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO skill_refinements
           (run_id, layer, old_skill_version_id, new_skill_version_id, refinement_summary, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, layer, old_skill_version_id, new_skill_version_id, refinement_summary, _now_iso()),
    )
    conn.commit()
    refinement_id = int(cursor.lastrowid)
    conn.close()
    return refinement_id


def insert_postmortem(
    layer: str,
    experiment_count: int,
    analysis: Optional[str] = None,
    changes_made: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO postmortems
           (layer, experiment_count, analysis, changes_made, trace_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (layer, experiment_count, analysis, changes_made, trace_path, _now_iso()),
    )
    conn.commit()
    pm_id = int(cursor.lastrowid)
    conn.close()
    return pm_id


def insert_results_log(
    layer: str,
    run_id: str,
    metric_value: Optional[float],
    metric_name: str,
    status: str,
    description: Optional[str] = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO results_log
           (layer, run_id, metric_value, metric_name, status, description, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (layer, run_id, metric_value, metric_name, status, description, _now_iso()),
    )
    conn.commit()
    conn.close()


def get_experiments_for_layer(layer: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT e.*, sv.version AS skill_version_label
           FROM experiments e
           LEFT JOIN skill_versions sv ON sv.id = e.skill_version_id
           WHERE e.layer = ?
           ORDER BY e.created_at""",
        (layer,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_best_experiment(layer: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM experiments
           WHERE layer = ? AND status IN ('success', 'keep') AND accuracy IS NOT NULL
           ORDER BY accuracy DESC
           LIMIT 1""",
        (layer,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_skill_lineage(layer: str, skill_name: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM skill_versions
           WHERE layer = ? AND skill_name = ?
           ORDER BY created_at""",
        (layer, skill_name),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
