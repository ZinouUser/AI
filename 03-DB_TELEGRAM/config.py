"""
config.py
─────────
Central configuration for the Smart School Clinic AI — Amal (أمل).

Flat structure (pedagogy-friendly). Will migrate to locales/ in Sprint 3.

Variable names : English
Logs           : Arabic (Qatari dialect)
Telegram       : Arabic (Qatari dialect)
HDMI display   : Arabic literary (فصحى)
Medical report : Arabic literary
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Robot & staff identity ─────────────────────────────────────────────────────
AI_NURSE_NAME = "أمل"           # AI robot name — fixed for all deployments
NURSE_NAME    = "نورة"          # Human nurse — change per school
SCHOOL_NAME   = "مدرسة النور"   # School name  — change per deployment

# ── Operation mode ─────────────────────────────────────────────────────────────
# "sim"  → PC / demo (no physical robot required)
# "real" → TonyPi Pro physical robot
MODE = os.getenv("MODE", "sim")

# ── API keys (loaded from .env — never hard-code) ─────────────────────────────
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
AZURE_SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Speech / Language ──────────────────────────────────────────────────────────
STT_LANGUAGE    = "ar-QA"       # Azure STT locale
TTS_MODEL       = "eleven_v3"   # ElevenLabs model
REPORT_LANGUAGE = "ar"          # Language for written medical reports

# ── Urgency levels (shared between SQLite and ChromaDB metadata) ───────────────
URGENCY_ROUTINE  = "routine"
URGENCY_WATCH    = "surveiller"
URGENCY_URGENT   = "urgent"
URGENCY_CRITICAL = "critique"

# ── Robot spoken messages — Qatari dialect ────────────────────────────────────
WELCOME_MSG  = f"أَهْلاً، أَنَا {AI_NURSE_NAME}. وِشْ فِيجْ اَلْيُومْ؟"
APPROVAL_MSG = f"بَانْتِظَارْ مُوَافَقَةْ {NURSE_NAME}... تَفَضَّلِي عِنْدْ جَاهِزِيَّتِجْ."
STANDBY_MSG  = f"أَنَا {AI_NURSE_NAME}، اَلْمُسَاعِدَةْ اَلذَّكِيَّةْ لِعِيَادَتِجْ. تَفَضَّلِي."

# ── Nurse approval keyword (موافقة نورة mechanism) ────────────────────────────
# Nurse speaks this word aloud to approve Amal's 3-step action plan
APPROVAL_KEYWORD = "تفضلي"

# ── Telegram notification categories (used in telegram_notif.py) ──────────────
NOTIF_NURSE_ROUTINE  = "nurse_routine"   # normal visit — nurse informed
NOTIF_NURSE_URGENT   = "nurse_urgent"    # urgent case — nurse alerted immediately
NOTIF_ADMINISTRATION = "administration" # principal / school admin
NOTIF_PARENTS        = "parents"        # parents / emergency contact
