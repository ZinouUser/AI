"""
voice_agent_llm.py
──────────────────
Amal Voice Agent — Azure STT + Gemini Flash LLM + ElevenLabs TTS
Phase 03 — DB + RAG + Identification

Usage:
    python voice_agent_llm.py --mode sim
    python voice_agent_llm.py --mode real
"""

import os
import sys
import re
import sqlite3
import threading
import argparse
import warnings
import tempfile
import subprocess
from difflib import SequenceMatcher
import time

import arabic_reshaper
from bidi.algorithm import get_display
import azure.cognitiveservices.speech as speechsdk
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from dotenv import load_dotenv

from gemini_brain import GeminiBrain, SYSTEM_PROMPT
from qatari_dialect import QatariDialect
from tashkil_display import TashkilDisplay
from telegram_notif import (
    notify_nurse_urgent,
    notify_school,
    notify_parents,
    notify_nurse_routine,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_patients import init_db, patient_context, log_visit, DB_PATH
from db_protocols import init_protocols, search_protocols, protocol_context

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

SPEECH_KEY          = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION       = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL    = "eleven_v3"
GEMINI_KEY          = os.getenv("GEMINI_API_KEY")

# ── Identification constants ───────────────────────────────────────────────────
MAX_ATTEMPTS    = 3
CLASS_ATTEMPTS  = 3
FUZZY_THRESHOLD = 0.60
MIN_TOKEN_SCORE = 0.55
NURSE_NAME      = "نورة"
AI_NAME         = "أمل"
MAX_TURNS       = 4
MIN_TURNS_BEFORE_END = 3   # Safety net: ignore END TAG before turn 3
STT_RETRIES = 3   # Max listen attempts when child response is unclear


eleven  = ElevenLabs(api_key=ELEVENLABS_KEY)
dialect = QatariDialect()
display = TashkilDisplay()
display.enabled = True

EXIT_KEYWORDS = ["وداعاً", "باي", "إنهاء", "أوقف", "خروج", "مع السلامة", "انتهى"]

# ── Close signal detection ─────────────────────────────────────────────────────
# Normalized at check time → hamza variants (إ/أ/ا) all match

# _URGENT_CLOSE_SIGNALS = [
#     "نورة راح تيجي", "نورة رح تيجي",
#     "الممرضة راح تيجي", "الممرضة جاية",
#     "بتواصل مع الممرضة", "بنبلغ الممرضة",
# ]

# _NORMAL_CLOSE_SIGNALS = [
#     "أرجعي للفصل", "ارجعي للفصل",
#     "روحي للفصل", "روحي الفصل",
#     "مع السلامة", "الله يشفيجْ", "الله يشفيج",
# ]

# _ROUTINE_NURSE_MODIFIERS = [
#     "في الفصل", "بالفصل", "عند فصلج", "بفصلها",
#     "لما تفرغ", "لما تخلص", "وقت الفرصة",
# ]

# _CLOSE_SIGNALS = _URGENT_CLOSE_SIGNALS + _NORMAL_CLOSE_SIGNALS

# ── Greeting detection (standby mode) ─────────────────────────────────────────
_GREETING_PATTERNS = [
    "السلام عليكم", "سلام عليكم", "السلام",
    "سلام", "اهلا", "مرحبا",
    "صباح الخير", "مساء الخير", "صباح", "مساء",
    # dialecte qatari / du Golfe
    "هلا", "هلو", "هالا",
    "حياك", "حياكم", "الله يحييك", "الله يحييج",
    "يا هلا", "يهلا",
]


# ── HDMI display simulation ────────────────────────────────────────────────────

def hdmi_display(message: str, delay: float = 0.03) -> None:
    """
    Affichage HDMI avec effet typewriter (lettre par lettre).
    delay : secondes entre chaque caractère (0.03 = ~30ms)
    """
    border = "▓" * 70
    print(f"\n\n{border}")
    print("  HDMI Display :\n")
    for line in message.strip().split("\n"):
        if not line.strip():
            print()
            continue
        processed = ar(line)   # toujours ar() — RTL garanti même lignes mixtes
        print("    ", end="", flush=True)
        for char in processed:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()
    print(f"\n{border}\n\n")



def _is_greeting(text: str) -> bool:
    t = normalize_ar(text)
    return any(normalize_ar(g) in t for g in _GREETING_PATTERNS)


def standby_listen(cfg: speechsdk.SpeechConfig) -> None:
    """Standby: loop on listen() until a greeting is detected."""
    print("\n" + "═" * 55)
    print(f"  [Standby] {ar('في انتظار الطالبة القادمة — قولي سلام ...')}")
    print("═" * 55)
    while True:
        text = listen(cfg)
        if not text:
            continue
        if _is_greeting(text):
            print(f"  [Standby] {ar('تم الكشف عن تحية')} ← '{ar(text)}'")
            return
        print(f"  [Standby] {ar('ليست تحية، ما زلت أنتظر')} ← '{ar(text)}'")


# ── STT pre-build state ────────────────────────────────────────────────────────
_stt_cfg: speechsdk.SpeechConfig | None = None
_next_recognizer: speechsdk.SpeechRecognizer | None = None


# ── Arabic display ─────────────────────────────────────────────────────────────

def ar(text: str) -> str:
    """Reshape + bidi for inline Arabic in f-strings."""
    return get_display(arabic_reshaper.reshape(text))


# ── STT ───────────────────────────────────────────────────────────────────────

def build_stt_config() -> speechsdk.SpeechConfig:
    global _stt_cfg
    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY missing in .env")
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "8000"
    )
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000"
    )
    _stt_cfg = cfg
    return cfg


def _is_arabic(text: str) -> bool:
    """Return True if at least 30% of non-space chars are Arabic."""
    chars = text.replace(' ', '')
    if not chars:
        return False
    return len(re.findall(r'[\u0600-\u06FF]', chars)) / len(chars) >= 0.30

def _has_latin_words(text: str) -> bool:
    """True si le texte contient au moins un mot latin (≥2 chars consécutifs)."""
    return bool(re.search(r'[a-zA-Z]{2,}', text))

def _prebuild_recognizer():
    """Pre-build next recognizer in background thread during TTS playback."""
    global _next_recognizer, _stt_cfg
    if _stt_cfg is None:
        return
    _auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["ar-QA", "ar-SA", "ar-AE"]
    )
    _next_recognizer = speechsdk.SpeechRecognizer(
        speech_config=_stt_cfg,
        auto_detect_source_language_config=_auto_detect
    )


# ── Phonetic recovery : Azure English → Arabic ─────────────────────────────
_PHONETIC_RECOVERY = {
    # يوم (youm = un jour)
    "yo"            : "يوم",
    "yom"           : "يوم",
    "you"           : "يوم",
    "yoom"          : "يوم",
    "min ion"       : "يوم",

    # يومين (yomeen = deux jours)
    "you mean"      : "يومين",
    "you men"       : "يومين",
    "yo mean"       : "يومين",
    "yo men"        : "يومين",
    "your man"      : "يومين",
    "your men"      : "يومين",
    "go man"        : "يومين",
    "in your men"   : "يومين",
    "you re men"    : "يومين",
    "youre men"     : "يومين",
    "menu man"      : "يومين",
    "menu. man"     : "يومين",
    "and young"     : "يومين",
    "min you mean"  : "يومين",
    "mean"          : "يومين",

    # من يومين (men yomeen = depuis deux jours)
    "min or m ate"  : "من يومين",
    "men you mean"  : "من يومين",
    "menu man men"  : "من يومين",
}


def _recover_arabic(text: str) -> str:
    """Recover Arabic from Azure phonetic English misrecognitions."""
    key = re.sub(r"[.,'!?؟،\-]", "", text.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    recovered = _PHONETIC_RECOVERY.get(key, text)
    if recovered != text:
        print(f"  [Recovery] '{text}' → '{ar(recovered)}'")
    return recovered


def listen(cfg: speechsdk.SpeechConfig) -> str:
    global _next_recognizer

    if _next_recognizer is not None:
        recognizer = _next_recognizer
        _next_recognizer = None
    else:
        _auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["ar-QA", "ar-SA", "ar-AE"]
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=cfg,
            auto_detect_source_language_config=_auto_detect
        )

    print("\n  Listening...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = _recover_arabic(result.text)
        if not _is_arabic(text):
            print(f"  [STT] Non-Arabic discarded: '{result.text}'")
            return ""
        display.print_arabic("  User : ", text)
        return text
    if result.reason == speechsdk.ResultReason.NoMatch:
        print("  WARNING: Nothing recognized.")
        return ""
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"  ERROR STT canceled: {details.reason} — {details.error_details}")
        return ""
    return ""


# ── TTS ───────────────────────────────────────────────────────────────────────

def speak(text: str) -> bool:
    if not ELEVENLABS_KEY or not ELEVENLABS_VOICE_ID:
        raise EnvironmentError("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID missing in .env")

    # 1. Tashkil check
    display.show_tashkil("Gemini tashkil", text)

    # 2. Dialect pipeline for display only
    display_text = dialect.process(text)
    display.print_arabic("  Amal : ", display_text)

    # 3. TTS text = raw Gemini with full tashkil
    tts_text = text.replace("\u06AF", "\u0642")
    tts_text = tts_text.rstrip() + " ."
    print(f"  TTS reçoit : {ar(tts_text)}")

    # 4. ElevenLabs call
    audio_generator = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=tts_text,
        model_id=ELEVENLABS_MODEL,
        language_code="ar",
    )
    audio_bytes = b"".join(audio_generator)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        mp3_path = tmp.name

    # ── OPTION A : Ubuntu natif / TonyPi Pro ──────────────────────────────
    # subprocess.run(["mpg123", "-q", mp3_path],
    #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # if os.path.exists(mp3_path): os.unlink(mp3_path)

    # ── OPTION B : WSL (actif en développement PC) ────────────────────────
    wav_path = mp3_path.replace(".mp3", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-af", "apad=pad_dur=0.5", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(["aplay", "-q", wav_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #time.sleep(0.4)
    # ── Safe unlink: ffmpeg may fail on minimal text → wav never created ──
    if os.path.exists(mp3_path):
        os.unlink(mp3_path)
    if os.path.exists(wav_path):
        os.unlink(wav_path)

    # Pre-build next recognizer in background while user prepares to speak
    threading.Thread(target=_prebuild_recognizer, daemon=True).start()

    return True


# ── Arabic normalization ───────────────────────────────────────────────────────

def normalize_ar(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    return text.strip()


_PHONETIC_MAP = str.maketrans({
    'ص': 'س', 'ث': 'س', 'ط': 'ت', 'ض': 'د',
    'ظ': 'ز', 'ذ': 'ز', 'ح': 'ه', 'ق': 'ك',
    'غ': 'خ', 'ع': 'ا',
})


def phonetic_ar(text: str) -> str:
    return normalize_ar(text).translate(_PHONETIC_MAP)


# ── Class normalization ────────────────────────────────────────────────────────

_GRADE_MAP = {
    "أول":"1","ثاني":"2","ثالث":"3","رابع":"4","خامس":"5","سادس":"6",
    "سابع":"7","ثامن":"8","تاسع":"9","عاشر":"10",
    "حادي عشر":"11","ثاني عشر":"12",
    "واحد":"1","اثنين":"2","اثنان":"2","ثلاثة":"3","ثلاث":"3",
    "اربعة":"4","أربعة":"4","خمسة":"5","ستة":"6","سبعة":"7",
    "ثمانية":"8","ثماني":"8","تسعة":"9","تسع":"9",
    "عشرة":"10","عشر":"10","أحد عشر":"11","احد عشر":"11",
    "اثنا عشر":"12","اثني عشر":"12",
    "١":"1","٢":"2","٣":"3","٤":"4","٥":"5",
    "٦":"6","٧":"7","٨":"8","٩":"9",
    "١٠":"10","١١":"11","١٢":"12",
}
_SECTION_MAP = {
    "أ":"أ","الف":"أ","ألف":"أ","آلف":"أ","a":"أ",
    "ب":"ب","باء":"ب","بي":"ب","b":"ب",
    "ج":"ج","جيم":"ج","c":"ج",
    "د":"د","دال":"د","d":"د",
}
_CLASS_STOPWORDS = {"الصف","صف","رقم","فصل"}


def normalize_class(text: str) -> str:

    # U+0621-U+064A = Arabic letters only (ء→ي), excludes punctuation (؟،؛ etc.)
    # U+0660-U+0669 = Arabic-Indic digits (٠-٩)
    text = re.sub(r'[^\u0621-\u064A\u0660-\u0669\s0-9a-zA-Z]', ' ', text)

    text = normalize_ar(text).lower()
    for sw in _CLASS_STOPWORDS:
        text = text.replace(normalize_ar(sw), "")
    text = text.strip()
    grade = ""
    section = ""
    for word, num in sorted(_GRADE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if normalize_ar(word) in text:
            grade = num
            text = text.replace(normalize_ar(word), "").strip()
            break
    if not grade:
        m = re.search(r'\d+', text)
        if m:
            grade = m.group()
            text = text[:m.start()] + text[m.end():]
    text_tokens = text.split()
    for word, letter in sorted(_SECTION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if normalize_ar(word) in text_tokens:
            section = letter
            break
    result = f"{grade}{section}"
    print(f"  [Class] '{ar(text)}' → '{ar(result)}'")
    return result


# ── Fuzzy name matching ────────────────────────────────────────────────────────

def _token_score(query: str, stored: str) -> tuple[float, float]:
    q_tok = phonetic_ar(query).split()
    s_tok = phonetic_ar(stored).split()
    if not q_tok or not s_tok:
        return 0.0, 0.0
    scores = [max(SequenceMatcher(None, qt, st).ratio() for st in s_tok)
              for qt in q_tok]
    return sum(scores) / len(scores), min(scores)


def fuzzy_identify_by_name(full_name: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM patients").fetchall()]
    conn.close()

    best, best_score = None, 0.0
    print(f"  [Fuzzy] Query: '{ar(phonetic_ar(full_name))}'")
    for p in rows:
        avg, worst = _token_score(full_name, p['full_name'])
        score = avg if worst >= MIN_TOKEN_SCORE else 0.0
        print(f"  [Fuzzy]   '{ar(phonetic_ar(p['full_name']))}' "
              f"avg={avg:.2f} min={worst:.2f} → {score:.2f}")
        if score > best_score:
            best_score, best = score, p

    if best and best_score >= FUZZY_THRESHOLD:
        display.print_arabic("Fuzzy ✓ ", f"{best['full_name']}  score={best_score:.2f}")
        return best
    print(f"  [Fuzzy] ✗ No match (best={best_score:.2f})")
    return None


_NAME_PREFIXES = [
    "أنا اسمي","أنا إسمي","اسمي","إسمي","أنا",
    "اسمه","إسمه","اسمها","إسمها","اسم","إسم",
]


def clean_name_from_stt(stt_text: str) -> str:
    text = stt_text.strip().rstrip(".")
    # Strip name prefixes
    for prefix in sorted(_NAME_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # Drop any token that contains no Arabic character (English noise, STT artifacts)
    tokens = text.split()
    arabic_only = [t for t in tokens if re.search(r'[\u0600-\u06FF]', t)]
    text = " ".join(arabic_only)
    display.print_arabic("Name ", text)
    return text


# ── Yes/no classifier ─────────────────────────────────────────────────────────

_QUICK_YES = {
    "صح","صحيح","آه","اه","أه","إيه","ايه",
    "نعم","أيوا","ايوا","أيوه","ايوه",
    "أكيد","اكيد","طبعاً","طبعا","بالتأكيد",
    "yes","oui","ok","okay",
}
_QUICK_NO = {"لا","لأ","لأه","لاه","ما","مو","no","non"}


def is_affirmative(text: str) -> bool | None:
    clean = re.sub(r'[\u060C\u061B\u061F\u06D4،؛؟!\?\.]+', '', text).strip()
    clean = normalize_ar(clean).lower()
    if clean in _QUICK_YES:
        print(f"  [isAffirm] YES (fast-path) ← '{ar(text)}'")
        return True
    if clean in _QUICK_NO:
        print(f"  [isAffirm] NO  (fast-path) ← '{ar(text)}'")
        return False
    client = genai.Client(api_key=GEMINI_KEY)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Classify: agreement or disagreement?\n"
            "Examples: 'yes'→YES 'صح'→YES 'آه'→YES 'no'→NO 'لا'→NO\n"
            "Reply ONE word: YES or NO or UNCLEAR\n"
            f"Answer: {clean}"
        ),
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
    )
    result = resp.text.strip().upper().split()[0] if resp.text.strip() else "UNCLEAR"
    print(f"  [isAffirm] {result} (Gemini) ← '{ar(text)}'")
    return True if result == "YES" else (False if result == "NO" else None)


def gemini_normalize_class(spoken: str) -> str:
    client = genai.Client(api_key=GEMINI_KEY)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Extract grade+section. Reply: number + Arabic letter only.\n"
            "Examples: 'التاسع ألف'→9أ 'العاشر باء'→10ب 'تسعة ألف'→9أ\n"
            f"Phrase: {spoken}\nReply:"
        ),
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
    )
    result = resp.text.strip().split()[0] if resp.text.strip() else ""
    print(f"  [Gemini Class] '{ar(result)}' ← '{ar(spoken)}'")
    return result


# ── 3-step identification ──────────────────────────────────────────────────────

def verify_class(candidate: dict, cfg: speechsdk.SpeechConfig) -> bool:
    attempt = 0
    while attempt < CLASS_ATTEMPTS:
        attempt += 1
        print(f"\n  [Step 2] Attempt {attempt}/{CLASS_ATTEMPTS}")
        if attempt == 1:
            speak("وِشْ صَفِّجْ؟")
        answer = listen(cfg)
        if not answer:
            # STT empty (non-Arabic discarded) — don't count this attempt
            attempt -= 1
            continue
        norm_given  = normalize_class(answer)
        norm_stored = normalize_class(candidate['class_code'])
        if norm_given and not re.search(r'[\u0600-\u06FF]', norm_given):
            print(f"  [Step 2] No section — Gemini fallback")
            norm_given = gemini_normalize_class(answer)
        if norm_given and norm_given == norm_stored:
            print(f"  [Step 2] ✓ '{ar(norm_given)}' == '{ar(norm_stored)}'")
            return True
        print(f"  [Step 2] ✗ '{ar(norm_given)}' ≠ '{ar(norm_stored)}'")
        if attempt < CLASS_ATTEMPTS:
            speak("رَقَمْ اَلصَّفّْ مَا طَابَقْ. عَادْ قُولِي رَقَمْ اَلصَّفّْ وَٱلشُّعْبَةْ.")
    speak(f"ما أَگْدَرْ أكمل. الصَّفّْ ما طَابَقّْ السِّجِل. "
          f"بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
    return False


def final_confirmation(candidate: dict, cfg: speechsdk.SpeechConfig) -> bool:
    speak(
        f"إِنْتِي {candidate['full_name']}، "
        f"مِنْ الصَّفّْ {candidate['class_lib']}، "
        f"صَحّْ؟"
    )
    for attempt in range(1, CLASS_ATTEMPTS + 1):
        print(f"\n  [Step 3] Attempt {attempt}/{CLASS_ATTEMPTS}")
        answer = listen(cfg)
        if not answer:
            if attempt < CLASS_ATTEMPTS:
                speak("مَا سَمَعْتِجْ. قُولِي صَحّْ أَوْ لَا.")
            continue
        result = is_affirmative(answer)
        if result is True:
            print(f"  [Step 3] ✓ {ar(candidate['full_name'])} confirmed")
            return True
        if result is False:
            speak(f"زِيْنّْ. بتواصل مع الممرضة {NURSE_NAME} لتصحيح السِّجِل.")
            return False
        # UNCLEAR → retry
        if attempt < CLASS_ATTEMPTS:
            speak("مَا فْهِمْتِجْ زِيْنْ، عَادْ قُولِي صَحّْ أَوْ لَا.")
    speak(f"ما فهمتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
    return False

def identify_loop(cfg: speechsdk.SpeechConfig,
                  skip_first_question: bool = False) -> dict | None:
    candidate = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  [Step 1] Attempt {attempt}/{MAX_ATTEMPTS}")
        if attempt == 1 and skip_first_question:
            pass  # question already asked in the merged greeting speak()
        elif attempt == 1:
            speak("وِشْ إِسْمِجْ الكامل؟")
        else:
            speak("ما فهمتِجْ زين، عاد قولي إِسْمِجْ الكامل.")
        stt_text = listen(cfg)

        if not stt_text:
            continue
        full_name = clean_name_from_stt(stt_text)
        if not full_name:
            continue
        if len(full_name.split()) == 1:
            print(f"  [Step 1] Single token '{ar(full_name)}' — asking for full name")
            speak(f"{full_name} إِيِــشْ؟")
            stt2 = listen(cfg)
            if stt2:
                fn2 = clean_name_from_stt(stt2)
                if fn2:
                    full_name = fn2
            if len(full_name.split()) < 2:
                speak("عاد قولي إِسْمِجْ الكامل، الاسم والنسب.")
                continue
        candidate = fuzzy_identify_by_name(full_name)
        if candidate:
            break
        speak("ما لَقِيتِجْ في السِّجل. عاد قولي إِسْمِجْ الكامل.")

    if not candidate:
        speak(f"ما قدرتِ أتعرف عليجْ. "
              f"بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
        return None

    print(f"\n  [Step 2] Verifying class for {ar(candidate['full_name'])}")
    if not verify_class(candidate, cfg):
        return None

    print(f"\n  [Step 3] Final confirmation for {ar(candidate['full_name'])}")
    if not final_confirmation(candidate, cfg):
        return None

    return candidate



# ── Nurse arrival protocol (Sara critical case — Étapes 5→12) ─────────────────

def _is_nurse_arrival(text: str) -> bool:
    """
    Detects Noura's arrival via her first name root 'نور' only.
    A student says 'الممرضة' — never 'نورة وصلت يا أمل'.
    'نور' covers all STT variants: نورة / نورا / نوره / نور
    """
    t = normalize_ar(text)
    has_nurse_name = "نور" in t
    has_arrival    = any(kw in t for kw in ["وصل", "جيت", "هنا", normalize_ar(AI_NAME)])
    return has_nurse_name and has_arrival


def _generate_briefing(patient: dict, symptoms_log: list[str]) -> str:
    allergies  = patient.get('allergies', '').strip()
    chronic    = patient.get('chronic', '').strip()
    class_code = patient.get('class_code', '')

    # ── Phrase 0 — intro : nom + classe (Python) ──────────────────────────────
    s0 = (
        f"{NURSE_NAME} — "
        f"ٱلطَّالِبَةْ {patient['full_name']} "
        f"مِنِ ٱلصَّفّْ {class_code}."
    )

    # ── Phrase 2 — dossier médical (Python) ───────────────────────────────────
    # Guard : "لا يوجد" / "aucune" / "" → traité comme vide
    _empty = {"لا يوجد", "aucune", "none", "لا شيء", ""}
    allergies_real = allergies if allergies not in _empty else ""
    chronic_real   = chronic   if chronic   not in _empty else ""

    if allergies_real and chronic_real:
        s2 = f"بِمَلَفَّهْا: حَسَاسِيَّةْ {allergies_real} وَعِنْدَهَا {chronic_real}."
    elif allergies_real:
        s2 = f"بِمَلَفَّهْا: حَسَاسِيَّةْ {allergies_real}، مَا فِي أَمْرَاضْ مُزْمِنَةْ."
    elif chronic_real:
        s2 = f"بِمَلَفَّهْا: {chronic_real}، مَا فِي حَسَاسِيَّةْ مِـسْـجَّلَةْ."
    else:
        s2 = "مَا فِي حَسَاسِيَّةْ وَلَا أَمْرَاضْ مُزْمِنَةْ بِمَلَفَّهْا."

    # ── Phrases 1 et 2 — Gemini : faits + hypothèse (déclaratif, pas de questions) ──
    first_name = patient['full_name'].split()[0]   # "سارة"

    prompt = (
        f"⚠️ كُلّْ كَلِمَةْ فِي رَدِّكْ يَجِبْ أَنْ تَحْمِلْ تَشْكِيلْ كَامِلْ عَلَى كُلّْ حَرْفْ.\n"
        f"أَنْتِ أَمَلْ. أَجِيبِي بِجُمْلَتَيْنِ فَقَطْ بِٱللَّهْجَةِ ٱلْقَطَرِيَّةْ.\n"
        f"جُمَلْ تَقْرِيرِيَّةْ فَقَطْ — مَمْنُوعْ ٱلْأَسْئِلَةْ.\n"
        f"ٱسْتَخْدِمِي ٱلِٱسْمَ ٱلْأَوَّلَ فَقَطْ: {first_name}\n"
        f"مَا قَالَتْهُ: {' | '.join(symptoms_log)}\n\n"
        f"ٱلْجُمْلَةْ 1: مَا حَدَثْ بِٱلضَّبْطْ — تَقْرِيرِيَّةْ\n"
        f"  مِثَالْ: '{first_name} أَكَلَتْ جِبْنْ قَبَلْ سَاعَةْ وَعِنْدَهَا أَلَمْ فِي بَطَنَهَا.'\n"
        f"ٱلْجُمْلَةْ 2: فَرْضِيَّتِجْ كَمُسَاعِدَةْ طِبِّيَّةْ — تَقْرِيرِيَّةْ\n"
        f"  مِثَالْ: 'ٱلظَّاهِرْ تَفَاعُلْ مَعَ ٱللَّاكْتُوزْ.'\n"
        f"جُمْلَتَيْنِ فَقَطْ. لَا تَذْكُرِي ٱسْمِجْ. لَا أَسْئِلَةْ.\n"
        f"⚠️ مَمْنُوعْ ذِكْرُ ٱلِٱسْمِ ٱلْكَامِلِ أَوِ ٱلصَّفّْ — ٱلِٱسْمُ ٱلْأَوَّلُ فَقَطْ."
    )

    client = genai.Client(api_key=GEMINI_KEY)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    s1_s3 = resp.text.strip()

    # ── Assemblage : intro → faits → dossier médical → hypothèse ─────────────
    briefing = f"n{s1_s3}\n{s2}" # no {s0}\
    print(f"  [Briefing] {ar(briefing)}")
    return briefing




def nurse_arrival_protocol(cfg: speechsdk.SpeechConfig,
                            patient: dict,
                            symptoms_log: list[str],
                            briefing: str) -> None:
    """
    Triggered when Amal detected an urgency.
    briefing: pre-generated clean summary ready to speak to Noura.
    """
    first_name    = patient['full_name'].split()[0]
    class_code    = patient.get('class_code', patient.get('class_name', ''))
    allergies     = patient.get('allergies', '').strip()
    chronic       = patient.get('chronic', '').strip()
    medication    = patient.get('medication', '').strip()
    medication_en = patient.get('medication_en', '').strip()
    symptoms      = " | ".join(symptoms_log)

    _empty     = {"", "aucune", "none", "لا شيء", "لا يوجد"}
    _med_empty = {"", "aucune", "none", "لا شيء", "لا يوجد"}

    # ── Rapport : antécédents + cause probable ────────────────────────────────
    allergies_real = allergies if allergies not in _empty else ""
    chronic_real   = chronic   if chronic   not in _empty else ""

    medical_history = " | ".join(filter(None, [allergies_real, chronic_real])) or "لَا يُوجَدْ"

    probable_cause = (
        f"تَفَاعُلْ مَعَ {allergies_real}" if allergies_real
        else chronic_real if chronic_real
        else "مَجْهُولْ"
    )

    # ── Étape 5 — HDMI alert + Telegram ──────────────────────────────────────
    if allergies_real:
        urgency_label = "رد فعل تحسسي محتمل"
    elif chronic_real:
        urgency_label = f"حالة مزمنة نشطة — {chronic_real}"
    else:
        urgency_label = "حالة طارئة"

    hdmi_display(
        f"⚠ {urgency_label}\n"
        f"● جاري استدعاء الممرضة\n"
        f"\n"
        f"[Telegram urgent → {NURSE_NAME} :\n"
        f"  {patient['full_name']} {class_code}"
        f" — {urgency_label} — تعالي العيادة فوراً]"
    )
    notify_nurse_urgent(patient)

    # ── Étape 6 — Companion loop while waiting ────────────────────────────────
    companion_msg  = (
        "كِيفْ تِحِسِّينْ ٱلْحِينْ؟ خِذِي نَفَسْ بَطِيءْ — "
        f"ٱلْمُمَرِّضَةْ جَايَةْ ٱلْحِينْ — {AI_NAME} إِنْ شَاءَ ٱللَّهْ مَعَاجْ."
    )
    wait_start      = time.time()
    nurse_arrived   = False
    COMPANION_EVERY = 6
    HDMI_EVERY      = 3

    print(f"\n  [Nurse Wait] Companion loop — waiting for {NURSE_NAME} ...")

    for cycle in range(30):
        if cycle % HDMI_EVERY == 0:
            elapsed = int(time.time() - wait_start)
            hdmi_display(
                f"⏱ {elapsed:02d}s منذ الاستدعاء\n"
                f"الطالبة مستقرة"
            )
        if cycle % COMPANION_EVERY == 0 and cycle > 0:
            speak(companion_msg)
        nurse_text = listen(cfg)
        if nurse_text and _is_nurse_arrival(nurse_text):
            print(f"  [Nurse] Arrived ← '{ar(nurse_text)}'")
            nurse_arrived = True
            break

    # ── Étape 7 — Briefing ────────────────────────────────────────────────────
    nurse_intro = (
        f"{NURSE_NAME} — "
        f"ٱلطَّالِبَةْ {patient['full_name']} "
        f"مِنِ ٱلصَّفّْ {class_code}."
    )
    full_briefing_for_nurse = f"{nurse_intro}\n{briefing}"
    sentences = [s.strip() for s in re.split(r'(?<=[.؟!،])\s+', full_briefing_for_nurse) if s.strip()]
    speak("\n".join(sentences))
    listen(cfg)

    # ── Étape 8 — Nurse confirms diagnosis ────────────────────────────────────
    speak(f"تِوَافِقِينْ عَلَى هَالتَّشْخِيصْ يَا {NURSE_NAME}؟")
    confirm_text = ""
    for _attempt in range(STT_RETRIES):
        raw = listen(cfg)
        if not raw:
            if _attempt < STT_RETRIES - 1:
                speak(f"مَا سَمَعْتِجْ يَا {NURSE_NAME}، عَادْ قُولِي.")
            continue
        confirm_text = raw
        break

    confirmed = False
    if confirm_text:
        af = is_affirmative(confirm_text)
        confirmed = af is True or af is None
    if not confirmed:
        speak(f"زِيْنْ يَا {NURSE_NAME}، بِٱنْتِظَارْ تَوْجِيهَاتِجْ.")
        listen(cfg)
        return

    # ── Étape 8b — Medication from patient file ───────────────────────────────
    if medication and medication not in _med_empty:
        speak(
            f"{NURSE_NAME} — فِي مَلَفّْ {first_name}، "
            f"ٱلدَّوَاءْ ٱلْمُعْتَمَدْ مِنْ أَهْلَهَا: {medication}."
        )
    else:
        speak(
            f"{NURSE_NAME} — مَا فِي دَوَاءْ مَذْكُورْ فِي مَلَفّْ {first_name}."
        )
    listen(cfg)

    # ── Étape 9 — Protocol approval ───────────────────────────────────────────
    speak("حَسَبْ اَلْبْرُتُكُولْ اَلْمُعْتَمَدْ — هَـذَا اَللِّـي لَازِمْ أَسَوِّيهْ اَلْحِينْ:")
    speak("أَوَّلْ شَيْ، بَـخَبِّرْ إِدَارَةْ اَلْمِدْرِسَهْ بِاللِّـي صَارْ.")
    speak(f"اَلثَّانِي — بَـطَرِّشْ رِسَالَةْ لِأَهْلْ {first_name} وَأَطَمِّنْهُمْ.")
    speak(f"اَلثَّالِثْ — بَـسَجِّلْ اَلْحَادِثَةْ فِي مَلَفّْ {first_name} تِلْقَائِيًّا.")
    speak(f"مُوَافَقَةْ {NURSE_NAME}؟")

    _APPROVAL_WORDS = {
        normalize_ar(w) for w in {
            "تفضل","تفضلي","صح","صحيح",
            "آه","اه","أه","ايه","إيه",
            "نعم","ايوا","ايوه","أيوا","أيوه",
            "زين","زينه","تمام",
            "موافقة","موافق","وافقت",
            "يلا","سوي","سويها","اشتغلي","ok","okay",
        }
    }

    def _nurse_approved(text: str) -> bool:
        n = normalize_ar(text)
        if n in _APPROVAL_WORDS:
            return True
        return any(w in n for w in _APPROVAL_WORDS)

    approved = False

    for _attempt in range(STT_RETRIES):
        approval_text = listen(cfg)
        if not approval_text:
            if _attempt < STT_RETRIES - 1:
                speak(f"مَا سَمَعْتِجْ يَا {NURSE_NAME}، عَادْ قُولِي.")
            continue
        if _nurse_approved(approval_text):
            approved = True
            break
        af = is_affirmative(approval_text)
        if af is True:
            approved = True
            break
        if af is False:
            speak(f"زِيْنْ يَا {NURSE_NAME}، بِٱنْتِظَارْ تَوْجِيهَاتِجْ.")
            return

    # ── Retry if still no response ─────────────────────────────────────────────
    if not approved:
        speak(
            f"يَا {NURSE_NAME} — مَا سِمَعْتِجْ زِيْنْ. "
            f"مِحْتَاجَةْ مُوَافَقْتِجْ عَشَانْ أَكَمِّلْ. "
            f"قُولِي — مُوَافَقَةْ؟"
        )
        final_text = listen(cfg)
        if final_text and (_nurse_approved(final_text) or is_affirmative(final_text) is True):
            approved = True

    # ── Hard block if still no approval ───────────────────────────────────────
    if not approved:
        speak(
            f"مَا أَگْـدَرْ أَكَمِّلْ بِدُونْ مُوَافَقْتِجْ يَا {NURSE_NAME}. "
            f"اَلْبْرُتُكُولْ وَاقِفْ — اَلْقَرَارْ قَرَارِجْ إِنتِي."
        )
        print("  [Step 9] ✗ Nurse approval missing — protocol BLOCKED (Effet Waw)")
        return

    print("  [Step 9] ✓ Nurse approval received — executing protocol")
    # … Étape 10+ …


    # ── Étape 10 — Exécution progressive + Telegram ───────────────────────────
    from datetime import datetime
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    hdmi_display(
        f"⚙ جَارٍ تَنْفِيذُ ٱلْبُرُوتُوكُولْ...\n"
        f"\n"
        f"  ① جَارٍ إِبْلَاغُ إِدَارَةِ ٱلْمَدْرَسَةْ..."
    )
    notify_school(patient, briefing)
    hdmi_display(
        f"  ✓  تَمَّ إِبْلَاغُ إِدَارَةِ ٱلْمَدْرَسَةْ — {time_str}\n"
        f"\n"
        f"  ② جَارٍ إِرْسَالُ رِسَالَةٍ لِأَهْلِ {first_name}..."
    )
    notify_parents(patient, briefing)
    hdmi_display(
        f"  ✓  تَمَّ إِبْلَاغُ إِدَارَةِ ٱلْمَدْرَسَةْ — {time_str}\n"
        f"  ✓  تَمَّ إِرْسَالُ رِسَالَةٍ لِأَهْلِ {first_name} — {time_str}\n"
        f"\n"
        f"  ③ جَارٍ تَسْجِيلُ ٱلتَّقْرِيرِ ٱلطِّبِّيِّ..."
    )
    time.sleep(2.0)

    hdmi_display(
        f"══════════════════════════════════════════════════════════════════\n"
        f"              تَقْرِيرُ حَادِثَةٍ طِبِّيَّةْ — ٱلْعِيَادَةُ ٱلذَّكِيَّةْ\n"
        f"                  Generated by {AI_NAME} · Azure Qatar Central\n"
        f"══════════════════════════════════════════════════════════════════\n"
        f"\n"
        f"  اِسْمُ ٱلطَّالِبَةْ   :  {patient['full_name']}\n"
        f"  ٱلصَّفّْ             :  {patient.get('class_lib', class_code)}\n"
        f"  ٱلتَّارِيخْ          :  {date_str}\n"
        f"  ٱلْوَقَتْ            :  {time_str}\n"
        f"\n"
        f"  ── ٱلْأَعْرَاضْ ───────────────────────────────────────────────\n"
        f"  ٱلْمُبَلَّغُ عَنْهَا   :  {symptoms_log[0] if symptoms_log else '—'}\n"
        f"  ٱلسَّبَبُ ٱلْمُحْتَمَلْ :  {probable_cause}\n"
        f"  ٱلتَّارِيخُ ٱلْمَرَضِيّْ:  {medical_history}\n"
        f"  ٱلدَّوَاءْ ٱلْمُعْتَمَدْ :  {medication if medication not in _med_empty else '—'}\n"
        f"\n"
        f"  ── ٱلْإِجْرَاءَاتُ ٱلْمُتَّخَذَةْ ──────────────────────────────\n"
        f"  ✓  تَمَّ ٱسْتِدْعَاءُ ٱلْمُمَرِّضَة {NURSE_NAME} فَوْرًا\n"
        f"  ✓  تَمَّ تَأْكِيدُ ٱلتَّشْخِيصِ مِنْ قِبَلِ ٱلْمُمَرِّضَة {NURSE_NAME}\n"
        f"  ✓  تَمَّ إِبْلَاغُ إِدَارَةِ ٱلْمَدْرَسَةْ\n"
        f"  ✓  تَمَّ إِرْسَالُ رِسَالَةٍ لِأَهْلِ ٱلطَّالِبَةِ وَتَطْمِينُهُمْ\n"
        f"  ✓  تَمَّ تَطْبِيقُ ٱلْبُرُوتُوكُولِ ٱلْمُعْتَمَدْ\n"
        f"\n"
        f"  ── ٱلْحَالَةْ ─────────────────────────────────────────────────\n"
        f"  مُسْتَقِرَّةْ تَحْتَ إِشْرَافِ ٱلْمُمَرِّضَة {NURSE_NAME}\n"
        f"\n"
        f"══════════════════════════════════════════════════════════════════"
    )

    # ── Étape 11 — Inform nurse all done ──────────────────────────────────────
    speak(
        f"{NURSE_NAME} — خَبَرْتُ اَلْإِدَارَةْ — وَطَمَّنْتْ أَهْلْ {first_name} — وَسَجِّ لْتْ اَلْحَادِثَةْ بِمَلَفَّهَا."
        f"كِلْ شَيْ تَمَامْ إِنْ شَا الله."
    )

    # ── Étape 12 — Nurse thanks → exit ────────────────────────────────────────
    listen(cfg)
    speak("دَايِمًا فِي ٱلْخِدْمَةْ.")


# ── End tag extractor ──────────────────────────────────────────────────────────

def _extract_end_tag(reply: str) -> tuple[str, str | None]:
    """
    Gemini appends [END:ROUTINE] or [END:URGENT] when closing the session.
    Returns (clean_reply_without_tag, "ROUTINE" | "URGENT" | None).
    None means the consultation is still ongoing.
    """
    match = re.search(r'\[END:(ROUTINE|URGENT)\]', reply, re.IGNORECASE)
    if match:
        tag   = match.group(1).upper()
        clean = reply[:match.start()].rstrip(" ،.—\n")
        print(f"  [END TAG] {tag}")
        return clean, tag
    return reply, None


# ── Consultation loop ──────────────────────────────────────────────────────────

def consultation_loop(cfg: speechsdk.SpeechConfig, brain: GeminiBrain,
                      patient: dict):
    first_name   = patient['full_name'].split()[0]
    symptoms_log = []

    # Urgency-aware greeting
    if patient.get('urgency_level') == 'urgent':
        opening = f"زِيْنّْ {first_name}! عندِجْ ملف خاص عندنا. وِشْ فِيجْ اليوم؟"
    else:
        opening = f"زِيْنّْ {first_name}! وِشْ فِيجْ اليوم؟"

    speak(opening)
    brain.prime_opening()

    # First listen — spoken retry on noise
    user_text = ""
    for _retry in range(3):
        user_text = listen(cfg)
        if user_text:
            break
        if _retry < 2:
            speak("ما سمعتِجْ زين، عاد قولي من فضلِجْ.")
    if not user_text:
        speak(f"ما أقدر أسمعِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
        return

    symptoms_log.append(user_text)
    turn            = 0
    session_closing = False
    urgent_close    = False      # True → nurse_arrival_protocol() appelé en fin
    prev_reply_norm = ""
    repeat_count    = 0

    while turn < MAX_TURNS:

        # ── Guard: empty user_text → retry silently, turn NOT incremented ──────
        if not user_text:
            user_text = listen(cfg)
            if user_text:
                if any(kw in user_text for kw in EXIT_KEYWORDS):
                    break
                symptoms_log.append(user_text)
                continue
            speak(f"ما أقدر أسمعِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
            session_closing = True
            break

        turn += 1   # counted only when we have valid input

        if any(kw in user_text for kw in EXIT_KEYWORDS):
            break

        # RAG with accumulated symptoms context
        rag_query = " | ".join(symptoms_log)
        protos = protocol_context(rag_query, n_results=2)
        print(f"  [RAG] Turn {turn} — query: '{ar(rag_query[:60])}'")

        turns_left = MAX_TURNS - turn
        if turns_left <= 1:
            turn_ctx = (
                "\n⚠️ آخِرُ دَوْرٍ — يَجِبُ إِصْدَارُ [END:ROUTINE] أَوْ [END:URGENT] "
                "فِي هَذَا ٱلرَّدّْ. مَمْنُوعْ طَرْحُ أَيِّ سُؤَالٍ إِضَافِيٍّ."
            )
        elif turns_left <= 2:
            turn_ctx = (
                f"\n⚠️ تَبَقَّى {turns_left} أَدْوَارْ — يَجِبُ ٱتِّخَاذُ قَرَارٍ قَرِيبًا."
            )
        else:
            turn_ctx = ""

        brain.set_context(
            patient_ctx=patient_context(patient["id"]),
            protocol_ctx=protos + turn_ctx,
        )

        reply = brain.think(user_text)

        # Strip end tag before speaking — tag is for system only, not read aloud
        reply, end_tag = _extract_end_tag(reply)

        # Skip empty / punctuation-only replies
        if not reply or not reply.strip(".").strip():
            break

        # Loop detection: same reply twice → force nurse escalation
        reply_norm = normalize_ar(reply)
        if reply_norm == prev_reply_norm:
            repeat_count += 1
            if repeat_count >= 2:
                print("  [Agent] Loop detected — forcing nurse escalation")
                speak(f"راح أتواصل مع الممرضة {NURSE_NAME} عشان تيجي تساعدِجْ.")
                urgent_close    = True
                session_closing = True
                break
        else:
            repeat_count    = 0
            prev_reply_norm = reply_norm

        speak(reply)

        # ── LLM decision via end tag ───────────────────────────────────────────────
        if end_tag and turn < MIN_TURNS_BEFORE_END:
            # Gemini closed too early — ignore tag, let consultation continue
            print(f"  [END TAG] Ignored — turn {turn} < {MIN_TURNS_BEFORE_END} minimum")
            end_tag = None

        if end_tag == "URGENT":
            print("  [Agent] END:URGENT — generating briefing + nurse protocol")
            briefing = _generate_briefing(patient, symptoms_log)
            urgent_close    = True
            session_closing = True
            break
        elif end_tag == "ROUTINE":
            print("  [Agent] END:ROUTINE — normal session close")
            notify_nurse_routine(patient, diagnosis=reply)
            session_closing = True
            break

        # ── Listen for child's response before next turn ───────────────────────
        user_text = ""
        _attempt  = 0
        while _attempt < STT_RETRIES:
            _attempt += 1
            raw = listen(cfg)
            if not raw:
                _attempt -= 1   # STT vide (non-Arabic discarded) — ne compte pas
                continue
            if _has_latin_words(raw):
                print(f"  [STT] Mixed text detected: '{raw}' — asking to repeat")
                _attempt -= 1   # texte mixte — ne compte pas non plus
                speak("مَا فْهِمْتِجْ زِيْنْ، عَادْ قُولِي.")
                continue
            user_text = raw
            break
        if user_text:
            symptoms_log.append(user_text)


    # ── Post-loop actions ──────────────────────────────────────────────────────
    if session_closing and not urgent_close:
        speak(
            f"سَلَامْتِجْ يَا {first_name}، ٱللَّهْ مَعِجْ — "
            f"مَعَ ٱلسَّلَامَةْ، وَرَبِّي يِعَافِيجْ."
        )
    elif urgent_close:
        nurse_arrival_protocol(cfg, patient, symptoms_log, briefing)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amal voice agent")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    args = parser.parse_args()

    display.init_log()
    print(f"\n=== Amal | STT=Azure ar-QA | LLM=Gemini Flash | "
          f"TTS=ElevenLabs v3 | mode={args.mode} ===\n")
    print(f"  [{'REAL' if args.mode == 'real' else 'SIM'} MODE]\n")

    init_db()
    init_protocols()

    cfg   = build_stt_config()
    brain = GeminiBrain()
    session_count = 0

    # ── Outer loop: one iteration = one student ────────────────────────────────
    while True:
        session_count += 1
        brain.reset()

        print(f"\n{'═'*55}")
        print(f"  [Session #{session_count}]")
        print(f"{'═'*55}\n")

        speak("هَلَا وَاللَّهْ! أَنَا أَمَلْ، مُسَاعِدَتِجْ الذَّكِيَّةْ فِي الْعِيَادَةْ. وِشْ إِسْمِجْ الكامِلْ؟")

        patient = identify_loop(cfg, skip_first_question=True)
        if not patient:
            print(f"\n  [Session #{session_count}] Identification failed.")
        else:
            print(f"\n  [Agent] ✓ {ar(patient['full_name'])} | "
                  f"{patient['class_code']} | {patient['urgency_level']}\n")
            consultation_loop(cfg, brain, patient)

        print(f"\n  [Session #{session_count}] Complete — entering standby.\n")
        standby_listen(cfg)


if __name__ == "__main__":
    main()
