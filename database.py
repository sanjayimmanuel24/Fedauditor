import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def create_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def set_password(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE email = ?",
        (generate_password_hash(password), email)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated

def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        user_id, name, stored_hash = row
        if check_password_hash(stored_hash, password):
            return {"id": user_id, "name": name, "email": email}
    return None
