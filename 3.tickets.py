"""SQLite storage: tickets + interaction log (for analytics)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "helpdesk.db"
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["low", "medium", "high"]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TicketDB:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        self._init()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    user_name   TEXT NOT NULL,
                    issue       TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    priority    TEXT NOT NULL DEFAULT 'medium',
                    status      TEXT NOT NULL DEFAULT 'Open',
                    source      TEXT NOT NULL,
                    confidence  REAL
                )""")
            c.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  TEXT NOT NULL,
                    user_name   TEXT NOT NULL,
                    query       TEXT NOT NULL,
                    matched_id  INTEGER,
                    category    TEXT,
                    confidence  REAL,
                    mode        TEXT,
                    outcome     TEXT NOT NULL      -- answered | ticket | solved | not_solved
                )""")

    # ------------------------------------------------------------ tickets
    def create_ticket(self, user_name: str, issue: str, category: str, priority: str,
                      source: str, confidence: float | None = None) -> int:
        now = _now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tickets (created_at, updated_at, user_name, issue, category, priority, status, source, confidence)"
                " VALUES (?, ?, ?, ?, ?, ?, 'Open', ?, ?)",
                (now, now, user_name or "Anonymous", issue, category, priority, source, confidence),
            )
            return cur.lastrowid

    def list_tickets(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM tickets", ()
        if status and status != "All":
            q, args = q + " WHERE status = ?", (status,)
        q += " ORDER BY id DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def update_status(self, ticket_id: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._conn() as c:
            c.execute("UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
                      (status, _now(), ticket_id))

    # -------------------------------------------------------- interactions
    def log_interaction(self, user_name: str, query: str, matched_id: int | None, category: str | None,
                        confidence: float | None, mode: str, outcome: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO interactions (created_at, user_name, query, matched_id, category, confidence, mode, outcome)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), user_name or "Anonymous", query, matched_id, category, confidence, mode, outcome),
            )
            return cur.lastrowid

    def set_outcome(self, interaction_id: int, outcome: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE interactions SET outcome = ? WHERE id = ?", (outcome, interaction_id))

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            by_outcome = {r[0]: r[1] for r in c.execute(
                "SELECT outcome, COUNT(*) FROM interactions GROUP BY outcome")}
            tickets = c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            open_t = c.execute("SELECT COUNT(*) FROM tickets WHERE status IN ('Open','In Progress')").fetchone()[0]
            by_cat = {r[0]: r[1] for r in c.execute(
                "SELECT category, COUNT(*) FROM tickets GROUP BY category ORDER BY 2 DESC")}
        solved = by_outcome.get("solved", 0)
        not_solved = by_outcome.get("not_solved", 0)
        rated = solved + not_solved
        return {
            "total_queries": total,
            "outcomes": by_outcome,
            "tickets": tickets,
            "open_tickets": open_t,
            "tickets_by_category": by_cat,
            "resolution_rate": (solved / rated) if rated else None,
            "auto_handled_rate": ((total - by_outcome.get("ticket", 0)) / total) if total else None,
        }
