"""
db_patients.py
──────────────
SQLite — Student health profiles.

Schema convention (language-agnostic):
  columns without _en  → school language (Arabic)
  columns with    _en  → English (enrollment form)
"""

import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patients.db")


def init_db():
    """Create tables and seed 5 students if DB is empty."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id                INTEGER PRIMARY KEY,
        first_name        TEXT NOT NULL,
        last_name         TEXT NOT NULL,
        full_name         TEXT NOT NULL,
        first_name_en     TEXT NOT NULL,
        last_name_en      TEXT NOT NULL,
        full_name_en      TEXT NOT NULL,
        age               INTEGER,
        grade             INTEGER,
        class_code        TEXT,
        class_code_en     TEXT,
        class_lib         TEXT,
        class_lib_en      TEXT,
        allergies         TEXT,
        allergies_en      TEXT,
        chronic           TEXT,
        chronic_en        TEXT,
        medication        TEXT,
        medication_en     TEXT,
        emergency_contact TEXT,
        urgency_level     TEXT DEFAULT 'routine',
        notes             TEXT,
        notes_en          TEXT,
        last_visit        TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id   INTEGER,
        visit_date   TEXT,
        symptoms_ar  TEXT,
        action_ar    TEXT,
        protocol_id  TEXT,
        notified     INTEGER DEFAULT 0,
        FOREIGN KEY (patient_id) REFERENCES patients(id)
    )""")

    # Migration : add medication columns if they don't exist yet
    for col in ("medication", "medication_en"):
        try:
            c.execute(f"ALTER TABLE patients ADD COLUMN {col} TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass   # column already exists

    c.execute("SELECT COUNT(*) FROM patients")
    if c.fetchone()[0] == 0:
        _seed(c)

    conn.commit()
    conn.close()
    print(f"[DB] SQLite ready → {DB_PATH}")


def _seed(c):
    patients = [
        (
            1,
            "فاطمة", "القحطاني", "فاطمة القحطاني",
            "Fatima", "Al-Qahtani", "Fatima Al-Qahtani",
            14, 9, "9أ", "9A", "التاسع ألف", "Grade 9 — Section A",
            "لا يوجد", "none",
            "رشح خفيف", "mild cold",
            "", "",
            "+974-5551-0001", "routine",
            "طالبة هادئة", "Calm student",
            None,
        ),
        (
            2,
            "سارة", "المنصوري", "سارة المنصوري",
            "Sara", "Al-Mansouri", "Sara Al-Mansouri",
            15, 10, "10ب", "10B", "العاشر باء", "Grade 10 — Section B",
            "لاكتوز", "lactose",
            "لا يوجد", "none",
            "إِنْتُولِيرَانْ — 5 قَطَرَاتْ", "Intoleran — 5 drops",
            "+974-5551-0002", "monitor",
            "حساسية اللاكتوز", "Lactose intolerance",
            None,
        ),
        (
            3,
            "روضة", "الشمري", "روضة الشمري",
            "Rawdha", "Al-Shammari", "Rawdha Al-Shammari",
            13, 8, "8ج", "8C", "الثامن جيم", "Grade 8 — Section C",
            "لا يوجد", "none",
            "ربو خفيف", "mild asthma",
            "بخاخ الربو في الحقيبة", "Asthma inhaler in schoolbag",
            "+974-5551-0003", "urgent",
            "ربو خفيف", "Mild asthma",
            None,
        ),
        (
            4,
            "ليلى", "العمادي", "ليلى العمادي",
            "Layla", "Al-Amadi", "Layla Al-Amadi",
            16, 11, "11أ", "11A", "الحادي عشر ألف", "Grade 11 — Section A",
            "لا يوجد", "none",
            "سكري نوع 1", "type 1 diabetes",
            "قلم الأنسولين", "Insulin pen",
            "+974-5551-0004", "monitor",
            "تحمل قلم الأنسولين دائماً", "Always carries insulin pen",
            None,
        ),
        (
            5,
            "حصة", "الجابر", "حصة الجابر",
            "Hessa", "Al-Jaber", "Hessa Al-Jaber",
            14, 9, "9ب", "9B", "التاسع باء", "Grade 9 — Section B",
            "لا يوجد", "none",
            "إغماء متكرر", "recurrent syncope",
            "", "",
            "+974-5551-0005", "urgent",
            "تاريخ إغماء — مراقبة عند الوقوف المفاجئ",
            "Syncope history — monitor on sudden standing",
            None,
        ),
    ]
    c.executemany(
        """INSERT OR IGNORE INTO patients VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        patients,
    )


def get_patient_by_id(pid: int) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ?", (pid,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_patients() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM patients")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history(patient_id: int, limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM visits WHERE patient_id = ? ORDER BY visit_date DESC LIMIT ?",
        (patient_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_visit(patient_id: int, symptoms_ar: str, action_ar: str,
              protocol_id: str = "", notified: bool = False):
    """Record a visit and update last_visit."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO visits
           (patient_id, visit_date, symptoms_ar, action_ar, protocol_id, notified)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (patient_id, date.today().isoformat(),
         symptoms_ar, action_ar, protocol_id, int(notified)),
    )
    c.execute(
        "UPDATE patients SET last_visit = ? WHERE id = ?",
        (date.today().isoformat(), patient_id),
    )
    conn.commit()
    conn.close()


def patient_context(patient_id: int) -> str:
    """Return a text summary for LLM injection (Arabic)."""
    p = get_patient_by_id(patient_id)
    if not p:
        return ""
    lines = [
        f"الاسم: {p['full_name']} ({p['full_name_en']})",
        f"الصف: {p['class_lib']} ({p['class_code_en']})",
        f"العمر: {p['age']} سنة",
        f"الحساسية: {p['allergies']}",
        f"الدواء المعتمد: {p.get('medication', '') or 'لا يوجد'}",
        f"الأمراض المزمنة: {p['chronic']}",
        f"مستوى الطوارئ: {p['urgency_level']}",
        f"ملاحظات: {p['notes']}",
    ]
    hist = get_history(patient_id, limit=3)
    if hist:
        lines.append("آخر الزيارات:")
        for v in hist:
            lines.append(f"  {v['visit_date']} — {v['symptoms_ar']}")
    return "\n".join(lines)


if __name__ == "__main__":
    init_db()
