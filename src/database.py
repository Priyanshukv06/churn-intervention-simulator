from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


DB_PATH = Path("data/sessions.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the sessions table if it doesn't exist.
    Call once at app startup.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            customer_index      INTEGER,
            original_prob       REAL    NOT NULL,
            simulated_prob      REAL    NOT NULL,
            prob_delta          REAL    NOT NULL,
            contract_original   TEXT,
            contract_simulated  TEXT,
            tenure_original     INTEGER,
            tenure_simulated    INTEGER,
            monthly_charges_orig  REAL,
            monthly_charges_sim   REAL,
            internet_original   TEXT,
            internet_simulated  TEXT,
            payment_original    TEXT,
            payment_simulated   TEXT,
            notes               TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_simulation(
    customer_index: int,
    original_customer: dict,
    modified_customer: dict,
    original_prob: float,
    simulated_prob: float,
    notes: str = "",
) -> int:
    """
    Insert one simulation session into the DB.
    Returns the new row ID.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO simulation_sessions (
            timestamp,
            customer_index,
            original_prob,
            simulated_prob,
            prob_delta,
            contract_original,
            contract_simulated,
            tenure_original,
            tenure_simulated,
            monthly_charges_orig,
            monthly_charges_sim,
            internet_original,
            internet_simulated,
            payment_original,
            payment_simulated,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            customer_index,
            round(original_prob, 4),
            round(simulated_prob, 4),
            round(simulated_prob - original_prob, 4),
            original_customer.get("Contract"),
            modified_customer.get("Contract"),
            int(original_customer.get("tenure", 0)),
            int(modified_customer.get("tenure", 0)),
            float(original_customer.get("MonthlyCharges", 0)),
            float(modified_customer.get("MonthlyCharges", 0)),
            original_customer.get("InternetService"),
            modified_customer.get("InternetService"),
            original_customer.get("PaymentMethod"),
            modified_customer.get("PaymentMethod"),
            notes,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def fetch_all_sessions(limit: int = 200) -> pd.DataFrame:
    """Return all logged sessions as a DataFrame, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM simulation_sessions ORDER BY id DESC LIMIT {limit}",
        conn,
    )
    conn.close()
    return df


def delete_session(session_id: int) -> None:
    """Delete a single session by ID."""
    conn = get_connection()
    conn.execute("DELETE FROM simulation_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def clear_all_sessions() -> None:
    """Wipe all sessions (useful for demo resets)."""
    conn = get_connection()
    conn.execute("DELETE FROM simulation_sessions")
    conn.commit()
    conn.close()


def get_session_stats() -> dict:
    """Return summary stats for the history tab dashboard."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT
            COUNT(*)                        AS total_sessions,
            ROUND(AVG(original_prob), 4)    AS avg_original_risk,
            ROUND(AVG(simulated_prob), 4)   AS avg_simulated_risk,
            ROUND(AVG(prob_delta), 4)       AS avg_delta,
            SUM(CASE WHEN prob_delta < 0 THEN 1 ELSE 0 END) AS interventions_that_helped
        FROM simulation_sessions
    """)
    row = cursor.fetchone()
    conn.close()
    if row and row["total_sessions"] > 0:
        return dict(row)
    return {
        "total_sessions": 0,
        "avg_original_risk": 0,
        "avg_simulated_risk": 0,
        "avg_delta": 0,
        "interventions_that_helped": 0,
    }


if __name__ == "__main__":
    init_db()
    print("DB initialised at", DB_PATH)

    # Smoke test
    row_id = log_simulation(
        customer_index=42,
        original_customer={
            "Contract": "Month-to-month",
            "tenure": 4,
            "MonthlyCharges": 74.35,
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        },
        modified_customer={
            "Contract": "Two year",
            "tenure": 4,
            "MonthlyCharges": 74.35,
            "InternetService": "Fiber optic",
            "PaymentMethod": "Bank transfer (automatic)",
        },
        original_prob=0.72,
        simulated_prob=0.31,
        notes="Upgraded contract to Two year",
    )
    print("Logged session ID:", row_id)

    df = fetch_all_sessions()
    print(df)

    stats = get_session_stats()
    print("Stats:", stats)
