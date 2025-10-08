import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("SQLITE_DB", "relatorios_nf.db")

def init_memory():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            question TEXT,
            answer TEXT,
            created_at TEXT
        )
        """)
        conn.commit()

def save_qa(user, question, answer):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO qa_history(user, question, answer, created_at) VALUES(?,?,?,?)",
                  (user, question, answer, datetime.now().isoformat()))
        conn.commit()

def get_history(user, limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT question, answer, created_at FROM qa_history WHERE user=? ORDER BY id DESC LIMIT ?", (user, limit))
        return c.fetchall()

def get_history_filteredold(user, start_date=None, end_date=None, limit=50):
    query = "SELECT question, answer, created_at FROM qa_history WHERE user=?"
    params = [user]

    if start_date:
        query += " AND date(created_at) >= date(?)"
        params.append(start_date)
    if end_date:
        query += " AND date(created_at) <= date(?)"
        params.append(end_date)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(query, tuple(params))
        return c.fetchall()

def get_history_filtered(user=None, start_date=None, end_date=None, limit=50):
    query = "SELECT user, question, answer, created_at FROM qa_history WHERE 1=1"
    params = []

    if user:
        query += " AND user=?"
        params.append(user)

    if start_date:
        query += " AND date(created_at) >= date(?)"
        params.append(start_date)
    if end_date:
        query += " AND date(created_at) <= date(?)"
        params.append(end_date)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(query, tuple(params))
        return c.fetchall()

def get_all_users():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT user FROM qa_history ORDER BY user")
        return [row[0] for row in c.fetchall()]
