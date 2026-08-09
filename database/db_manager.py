"""
SQLite persistence layer for people (students/employees) and attendance records.
"""
import sqlite3
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DBManager:
    def __init__(self, db_path: str = config.DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id   TEXT PRIMARY KEY,
                    label_id    INTEGER UNIQUE,
                    name        TEXT NOT NULL,
                    person_type TEXT NOT NULL CHECK(person_type IN ('student', 'employee')),
                    department  TEXT,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT NOT NULL,
                    date        TEXT NOT NULL,
                    time        TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'present',
                    confidence  REAL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id),
                    UNIQUE(person_id, date)
                )
            """)

    def add_person(self, person_id: str, name: str, person_type: str, department: str = None) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT MAX(label_id) FROM persons")
            max_label = cur.fetchone()[0]
            label_id = 0 if max_label is None else max_label + 1
            conn.execute(
                "INSERT INTO persons (person_id, label_id, name, person_type, department, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, label_id, name, person_type, department, datetime.datetime.now().isoformat())
            )
            return label_id

    def get_all_persons(self):
        with self._connect() as conn:
            cur = conn.execute("SELECT person_id, label_id, name, person_type, department FROM persons")
            return cur.fetchall()

    def get_person_by_label(self, label_id: int):
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT person_id, name, person_type, department FROM persons WHERE label_id = ?",
                (label_id,)
            )
            return cur.fetchone()

    def person_exists(self, person_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM persons WHERE person_id = ?", (person_id,))
            return cur.fetchone() is not None

    def mark_attendance(self, person_id: str, confidence: float) -> bool:
        now = datetime.datetime.now()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO attendance (person_id, date, time, status, confidence) "
                    "VALUES (?, ?, ?, 'present', ?)",
                    (person_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), confidence)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def already_marked_today(self, person_id: str) -> bool:
        today = datetime.date.today().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM attendance WHERE person_id = ? AND date = ?", (person_id, today)
            )
            return cur.fetchone() is not None

    def get_attendance_by_date(self, date: str = None):
        date = date or datetime.date.today().isoformat()
        with self._connect() as conn:
            cur = conn.execute("""
                SELECT a.date, a.time, p.person_id, p.name, p.person_type, p.department, a.status, a.confidence
                FROM attendance a JOIN persons p ON a.person_id = p.person_id
                WHERE a.date = ?
                ORDER BY a.time
            """, (date,))
            return cur.fetchall()
