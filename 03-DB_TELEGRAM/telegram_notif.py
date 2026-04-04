"""
telegram_notif.py
─────────────────
Telegram notifications for Amal Smart Clinic
4 notification types:
  T1 — URGENT  : nurse called immediately
  T2 — SCHOOL  : school management informed
  T3 — PARENTS : parents reassured
  T4 — ROUTINE : nurse soft follow-up
"""

import os
import requests
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv

load_dotenv(override=True)

_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
_NURSE_ID   = os.getenv("TELEGRAM_NURSE_CHAT_ID", "")
_SCHOOL_ID  = os.getenv("TELEGRAM_SCHOOL_CHAT_ID", "")
_PARENTS_ID = os.getenv("TELEGRAM_PARENTS_CHAT_ID", "")

_API = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"

_EMPTY = {"", "لا يوجد", "aucune", "none", "لا شيء"}


def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def _case_info(patient: dict) -> tuple[str, str, str]:
    """
    Extract allergy / chronic / case_type from patient dict.
    case_type : 'allergy' | 'chronic' | 'general'
    """
    allergies = patient.get("allergies", "").strip()
    chronic   = patient.get("chronic",   "").strip()
    allergies = allergies if allergies not in _EMPTY else ""
    chronic   = chronic   if chronic   not in _EMPTY else ""
    if allergies:
        case_type = "allergy"
    elif chronic:
        case_type = "chronic"
    else:
        case_type = "general"
    return allergies, chronic, case_type


def _send(chat_id: str, text: str, label: str = "") -> bool:
    if not _TOKEN or not chat_id:
        print(f"  [Telegram] ⚠ {ar('TOKEN أو CHAT_ID مفقود — تم التخطي')}")
        return False
    try:
        resp = requests.post(_API, json={
            "chat_id"   : chat_id,
            "text"      : text,
            "parse_mode": "HTML",
        }, timeout=5)
        ok   = resp.status_code == 200
        dest = label if label else chat_id
        if ok:
            print(f"  [Telegram] ✓ {ar('تَمَّ الإِرْسَالُ إِلَى ' + dest)}")
        else:
            print(f"  [Telegram] ✗ {ar(f'خَطَأْ {resp.status_code} — {dest}')}")
        return ok
    except Exception as e:
        print(f"  [Telegram] ✗ {ar(f'اسْتِثْنَاءْ : {e}')}")
        return False



# ── T1 — URGENT : nurse called immediately ────────────────────────────────────

def notify_nurse_urgent(patient: dict) -> bool:
    full_name  = patient.get("full_name",  "—")
    class_code = patient.get("class_code", "—")
    allergies, chronic, case_type = _case_info(patient)

    if case_type == "allergy":
        case_lines = (
            f"الحالة   : رد فعل تحسسي محتمل\n"
            f"الحساسية : {allergies}\n"
        )
    elif case_type == "chronic":
        case_lines = (
            f"الحالة   : حالة مزمنة نشطة\n"
            f"المرض    : {chronic}\n"
        )
    else:
        case_lines = "الحالة   : حالة طارئة\n"

    text = (
        "🚨 <b>تنبيه عاجل — العيادة الذكية</b>\n\n"
        f"الطالبة : <b>{full_name}</b>\n"
        f"الصف     : {class_code}\n"
        f"{case_lines}"
        "\n⚠ <b>تعالي العيادة فوراً</b>"
    )
    return _send(_NURSE_ID, text, label="ٱلْمُمَرِّضَةْ نُورَةْ 🚨")


# ── T2 — SCHOOL : school management informed ──────────────────────────────────

def notify_school(patient: dict, symptoms: str) -> bool:
    from datetime import datetime
    now        = datetime.now()
    full_name  = patient.get("full_name",  "—")
    class_code = patient.get("class_code", "—")
    allergies, chronic, case_type = _case_info(patient)

    if case_type == "allergy":
        case_line = f"الحالة   : رد فعل تحسسي محتمل — حساسية {allergies}\n"
    elif case_type == "chronic":
        case_line = f"الحالة   : حالة مزمنة نشطة — {chronic}\n"
    else:
        case_line = "الحالة   : حالة طارئة\n"

    text = (
        "📋 <b>إشعار طبي — العيادة الذكية</b>\n\n"
        f"التاريخ  : {now.strftime('%Y-%m-%d')}\n"
        f"الوقت    : {now.strftime('%H:%M')}\n\n"
        f"الطالبة  : <b>{full_name}</b>\n"
        f"الصف     : {class_code}\n"
        f"{case_line}"
        f"التقييم  : {symptoms}\n\n"
        "الحالة تحت إشراف الممرضة — البروتوكول المعتمد تم تطبيقه."
    )
    return _send(_SCHOOL_ID, text, label="إِدَارَةْ ٱلْمَدْرَسَةْ")


# ── T3 — PARENTS : reassurance message ────────────────────────────────────────

def notify_parents(patient: dict, symptoms: str) -> bool:
    first_name = patient.get("full_name", "—").split()[0]
    class_code = patient.get("class_code", "—")
    allergies, chronic, case_type = _case_info(patient)

    # ── Medication check ──────────────────────────────────────────────────────
    _med_empty = {"", "aucune", "none", "لا شيء", "لا يوجد"}
    medication = patient.get("medication", "").strip()
    med_real   = medication if medication not in _med_empty else ""

    if med_real:
        note_line = f"💊 الدواء المعتمد من ملفها ({med_real}) تم إعطاؤه حسب البروتوكول.\n"
    else:
        note_line = "إذا عندها دواء خاص يرجى التواصل مع إدارة المدرسة.\n"

    if case_type == "allergy":
        context_line = f"راجعت العيادة اليوم بسبب تفاعل مع حساسية {allergies}.\n"
    elif case_type == "chronic":
        context_line = f"راجعت العيادة اليوم بسبب أعراض متعلقة بـ {chronic}.\n"
    else:
        context_line = f"راجعت العيادة الذكية اليوم بسبب : {symptoms}\n"

    text = (
        "💚 <b>رسالة من العيادة الذكية — مدرستكم</b>\n\n"
        "السلام عليكم،\n"
        f"بنتكم <b>{first_name}</b> من الصف {class_code} بخير إن شاء الله.\n\n"
        f"{context_line}"
        f"{note_line}"
        "ما في داعي للقلق — نبلغكم لو في أي تطور.\n\n"
        "🤖 <i>Generated by أمل · العيادة الذكية</i>"
    )
    return _send(_PARENTS_ID, text, label="أَهْلْ ٱلطَّالِبَةْ")


# ── T4 — ROUTINE : nurse soft follow-up ───────────────────────────────────────

def notify_nurse_routine(patient: dict, diagnosis: str = "") -> bool:
    full_name  = patient.get("full_name",  "—")
    class_code = patient.get("class_code", "—")
    text = (
        "🟡 <b>متابعة روتينية — العيادة الذكية</b>\n\n"
        f"الطالبة : <b>{full_name}</b>\n"
        f"الصف     : {class_code}\n\n"
        "راجعت العيادة الذكية ورجعت لفصلها.\n"
    )
    if diagnosis:
        text += f"\n🔍 <b>تقييم أمل</b> : {diagnosis}\n"
    text += "\nلو تقدرين تشوفينها في الفصل وقت الفرصة."
    return _send(_NURSE_ID, text, label="ٱلْمُمَرِّضَةْ نُورَةْ")


# Just for Test
if __name__ == "__main__":
    test_patient = {
        "full_name" : "سارة المنصوري",
        "class_code": "10ب",
        "allergies" : "لاكتوز",
    }
    notify_nurse_urgent(test_patient)
    notify_school(test_patient, "ألم في البطن")
    notify_parents(test_patient, "ألم في البطن")
    notify_nurse_routine(test_patient)
