"""
test_identify.py
────────────────
3-step student identification: name → class → final confirmation.
STT (Azure ar-QA) + ElevenLabs TTS + Gemini (yes/no + class fallback).

Step 1 — Full name
  STT → clean_name_from_stt() → single-token check → phonetic token fuzzy match → candidate
  Single token (first name only): speak "{name} ءَايِشْ؟" → re-listen → require 2+ tokens
  Both first name AND last name must individually score >= MIN_TOKEN_SCORE

Step 2 — Class verification (open question, no leading answer)
  "وش صفج؟" → STT → normalize_class() → Gemini fallback if no section → exact match
  Handles ordinals (التاسع), cardinals (تسعة), digits (9), Latin (9A)
  3 attempts — case-specific retry messages

Step 3 — Final identity confirmation (language-agnostic)
  Amal reads full name + class → student confirms
  is_affirmative() fast-path Gulf Arabic + Gemini fallback
  Only explicit YES proceeds to consultation

Normalization pipeline for names (both sides before comparison):
  raw → normalize_ar() : tashkil, alef variants, ta marbuta, ya
      → phonetic_ar()  : ص→س  ط→ت  ح→ه  ق→ك  ض→د  ظ/ذ→ز  غ→خ  ع→ا
      → token split    : order-independent
      → SequenceMatcher: tolerates remaining character-level errors
"""

import os
import sys
import re
import sqlite3
import tempfile
import subprocess
import warnings
from difflib import SequenceMatcher

import arabic_reshaper
from bidi.algorithm import get_display
import azure.cognitiveservices.speech as speechsdk
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_patients import init_db, patient_context, DB_PATH

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

AZURE_KEY    = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
ELEVEN_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID     = os.getenv("ELEVENLABS_VOICE_ID")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY")

MAX_ATTEMPTS    = 3     # retries for name recognition
CLASS_ATTEMPTS  = 3     # retries for class verification
FUZZY_THRESHOLD = 0.60  # minimum average token score to accept a match
MIN_TOKEN_SCORE = 0.55  # each name token must individually pass this
NURSE_NAME      = "نورة"

eleven = ElevenLabs(api_key=ELEVEN_KEY)


# ── Arabic display ─────────────────────────────────────────────────────────────

def print_ar(label: str, text: str):
    """Whole line is Arabic → reshape + bidi."""
    displayed = get_display(arabic_reshaper.reshape(text))
    print(f"  [{label}] {displayed}")


def ar(text: str) -> str:
    """Arabic embedded in an English f-string → reshape + bidi inline."""
    return get_display(arabic_reshaper.reshape(text))


# ── Orthographic normalization ─────────────────────────────────────────────────

def normalize_ar(text: str) -> str:
    """
    Normalize Arabic orthography:
      - Remove tashkil
      - Alef variants → ا   (أ إ آ ٱ → ا)
      - Ta marbuta    → ه   (ة → ه)
      - Ya variant    → ي   (ى → ي)
    """
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    return text.strip()


# ── Phonetic normalization ─────────────────────────────────────────────────────

_PHONETIC_MAP = str.maketrans({
    'ص': 'س',   # emphatic S   → S
    'ث': 'س',   # TH           → S
    'ط': 'ت',   # emphatic T   → T
    'ض': 'د',   # emphatic D   → D
    'ظ': 'ز',   # emphatic DH  → Z
    'ذ': 'ز',   # DH           → Z
    'ح': 'ه',   # pharyngeal H → H
    'ق': 'ك',   # uvular Q     → K
    'غ': 'خ',   # voiced uvular→ KH
    'ع': 'ا',   # pharyngeal   → A
})


def phonetic_ar(text: str) -> str:
    """Apply phonetic normalization after normalize_ar()."""
    return normalize_ar(text).translate(_PHONETIC_MAP)


# ── Class normalization ────────────────────────────────────────────────────────

_GRADE_MAP = {
    # Ordinals
    "أول":        "1",  "ثاني":       "2",  "ثالث":       "3",
    "رابع":       "4",  "خامس":       "5",  "سادس":       "6",
    "سابع":       "7",  "ثامن":       "8",  "تاسع":       "9",
    "عاشر":       "10", "حادي عشر":  "11", "ثاني عشر":   "12",
    # Cardinals
    "واحد":       "1",  "اثنين":      "2",  "اثنان":      "2",
    "ثلاثة":      "3",  "ثلاث":       "3",
    "اربعة":      "4",  "أربعة":      "4",
    "خمسة":       "5",
    "ستة":        "6",
    "سبعة":       "7",
    "ثمانية":     "8",  "ثماني":      "8",
    "تسعة":       "9",  "تسع":        "9",
    "عشرة":       "10", "عشر":        "10",
    "أحد عشر":   "11", "احد عشر":   "11",
    "اثنا عشر":  "12", "اثني عشر":  "12",
    # Arabic-Indic digits
    "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5",
    "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "١٠": "10", "١١": "11", "١٢": "12",
}

_SECTION_MAP = {
    "أ": "أ", "الف": "أ", "ألف": "أ", "آلف": "أ", "a": "أ",
    "ب": "ب", "باء": "ب", "بي":  "ب",              "b": "ب",
    "ج": "ج", "جيم": "ج",                           "c": "ج",
    "د": "د", "دال": "د",                           "d": "د",
}

_CLASS_STOPWORDS = {"الصف", "صف", "رقم", "فصل"}


def normalize_class(text: str) -> str:
    """
    Convert any spoken class variant to canonical "9أ" format.
    Strips punctuation first to avoid trailing dots breaking token matching.
    Section match is token-based (not substring) to avoid false positives.
    Returns "" if parsing fails.
    """
    # Strip punctuation before any processing
    text = re.sub(r'[^\u0600-\u06FF\s0-9a-zA-Z]', ' ', text)
    text = normalize_ar(text).lower()

    for sw in _CLASS_STOPWORDS:
        text = text.replace(normalize_ar(sw), "")
    text = text.strip()

    grade   = ""
    section = ""

    # Longest match first (handles "حادي عشر" before "عشر")
    for word, num in sorted(_GRADE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        norm_word = normalize_ar(word)
        if norm_word in text:
            grade = num
            text  = text.replace(norm_word, "").strip()
            break

    # Fallback: bare digit(s)
    if not grade:
        m = re.search(r'\d+', text)
        if m:
            grade = m.group()
            text  = text[:m.start()] + text[m.end():]

    # Token-based section match — avoids "ال" matching "ا" (normalized أ)
    text_tokens = text.split()
    for word, letter in sorted(_SECTION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        norm_word = normalize_ar(word)
        if norm_word in text_tokens:
            section = letter
            break

    result = f"{grade}{section}"
    print(f"  [Class] '{ar(text)}' → '{ar(result)}'")
    return result


# ── Token-based fuzzy score ────────────────────────────────────────────────────

def _token_score(query: str, stored: str) -> tuple[float, float]:
    """
    Order-independent phonetic fuzzy score.
    Returns (average, minimum) token scores.
    """
    q_tokens = phonetic_ar(query).split()
    s_tokens = phonetic_ar(stored).split()

    if not q_tokens or not s_tokens:
        return 0.0, 0.0

    scores = []
    for qt in q_tokens:
        best = max(SequenceMatcher(None, qt, st).ratio() for st in s_tokens)
        scores.append(best)

    return sum(scores) / len(scores), min(scores)


# ── DB fuzzy search ────────────────────────────────────────────────────────────

def fuzzy_identify_by_name(full_name: str) -> dict | None:
    """
    Search all patients by full name using token-based phonetic fuzzy match.

    Two conditions must BOTH be true to accept a match:
      1. Average token score >= FUZZY_THRESHOLD (0.60)
      2. Minimum token score >= MIN_TOKEN_SCORE  (0.55)

    Condition 2 prevents a correct first name from compensating
    a completely wrong last name (or vice versa).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM patients")
    all_patients = [dict(r) for r in c.fetchall()]
    conn.close()

    best, best_score = None, 0.0

    print(f"  [Fuzzy] Query phonetic: '{ar(phonetic_ar(full_name))}'")
    for p in all_patients:
        avg, worst = _token_score(full_name, p['full_name'])

        # Reject if any single token is too weak
        score = avg if worst >= MIN_TOKEN_SCORE else 0.0

        print(f"  [Fuzzy]   '{ar(phonetic_ar(p['full_name']))}' "
              f"avg={avg:.2f} min={worst:.2f} → {score:.2f}")

        if score > best_score:
            best_score, best = score, p

    if best and best_score >= FUZZY_THRESHOLD:
        print_ar("Fuzzy ✓", f"{best['full_name']}  score={best_score:.2f}")
        return best

    print(f"  [Fuzzy] ✗ No match (best={best_score:.2f})")
    return None


# ── Name extraction from STT ───────────────────────────────────────────────────

_NAME_PREFIXES = [
    "أنا اسمي", "أنا إسمي",
    "اسمي", "إسمي",
    "أنا",
    "اسمه", "إسمه", "اسمها", "إسمها",
]


def clean_name_from_stt(stt_text: str) -> str:
    """
    Remove common spoken prefixes → bare full name.
    Deterministic, offline, no LLM needed.
    """
    text = stt_text.strip().rstrip(".")
    for prefix in sorted(_NAME_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    print_ar("Name", text)
    return text


# ── TTS ───────────────────────────────────────────────────────────────────────

def speak(text: str):
    """Speak Arabic via ElevenLabs → ffmpeg → aplay (WSL-safe)."""
    print_ar("Amal", text)
    audio = b"".join(
        eleven.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text.rstrip() + " .",
            model_id="eleven_v3",
            language_code="ar",
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio)
        mp3 = f.name

    wav = mp3.replace(".mp3", ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", mp3, "-af", "apad=pad_dur=0.5", wav],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["aplay", "-q", wav],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(mp3)
    os.unlink(wav)


def _is_arabic(text: str) -> bool:
    """Return True if at least 50% of non-space characters are Arabic."""
    chars = text.replace(' ', '')
    if not chars:
        return False
    arabic = len(re.findall(r'[\u0600-\u06FF]', chars))
    return arabic / len(chars) >= 0.5


# ── STT ───────────────────────────────────────────────────────────────────────

def listen() -> str:
    """Capture one Arabic utterance via Azure STT (ar-QA).

    Discards result if recognized text is not predominantly Arabic
    (filters background noise mis-recognized as English).

    Timeouts (ms):
      InitialSilence : 8 000  — wait up to 8s for speech to start
      EndSilence     : 2 000  — wait 2s of silence after speech ends
    """
    cfg = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    cfg.speech_recognition_language = "ar-QA"
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "8000"
    )
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "2000"
    )
    rec = speechsdk.SpeechRecognizer(speech_config=cfg)
    print("  [STT] Listening...")
    result = rec.recognize_once_async().get()



    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        if not _is_arabic(result.text):
            print(f"  [STT] Non-Arabic discarded: '{result.text}'")
            return ""
        print_ar("STT", result.text)
        return result.text

    print(f"  [STT] Nothing recognized ({result.reason})")
    return ""
# ── Language-agnostic yes/no classifier ───────────────────────────────────────

_QUICK_YES = {
    "صح", "صحيح", "آه", "اه", "أه", "إيه", "ايه",
    "نعم", "أيوا", "ايوا", "أيوه", "ايوه",
    "أكيد", "اكيد", "طبعاً", "طبعا", "بالتأكيد",
    "yes", "oui", "ok", "okay",
}
_QUICK_NO = {
    "لا", "لأ", "لأه", "لاه", "ما", "مو", "لا ما",
    "no", "non",
}


def is_affirmative(text: str) -> bool | None:
    """
    Classify a short spoken answer as YES / NO / UNCLEAR.
    Fast path: Gulf Arabic keyword list.
    Fallback: Gemini (language-agnostic, max 5 tokens).
    """
    clean = re.sub(r'[\u060C\u061B\u061F\u06D4،؛؟!\?\.]+', '', text).strip()
    clean = normalize_ar(clean).lower()

    if clean in _QUICK_YES:
        print(f"  [isAffirm] YES (fast-path) ← '{ar(text)}'")
        return True
    if clean in _QUICK_NO:
        print(f"  [isAffirm] NO  (fast-path) ← '{ar(text)}'")
        return False

    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Classify the spoken answer below as agreement or disagreement.\n"
            "Examples:\n"
            "  'yes'  → YES\n"
            "  'oui'  → YES\n"
            "  'صح'   → YES\n"
            "  'آه'   → YES\n"
            "  'no'   → NO\n"
            "  'لا'   → NO\n"
            "Reply with exactly one word: YES or NO or UNCLEAR\n"
            f"Answer: {clean}"
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=5,
        ),
    )
    result = response.text.strip().upper()
    result = result.split()[0] if result.split() else "UNCLEAR"
    print(f"  [isAffirm] {result} (Gemini) ← '{ar(text)}'")

    if result == "YES":
        return True
    if result == "NO":
        return False
    return None


# ── Gemini class fallback ──────────────────────────────────────────────────────

def gemini_normalize_class(spoken: str) -> str:
    """
    Fallback: ask Gemini to extract class code from spoken Arabic.
    Returns canonical "9أ" format, or "" on failure.
    """
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Extract the school grade and section from this Arabic phrase.\n"
            "Reply ONLY with the grade number followed by the Arabic section letter.\n"
            "Section letters: أ=alef/ألف, ب=baa/باء, ج=jeem/جيم, د=dal/دال\n"
            "Examples:\n"
            "  'التاسع ألف'     → 9أ\n"
            "  'العاشر باء'     → 10ب\n"
            "  'الثامن جيم'    → 8ج\n"
            "  'الحادي عشر أ'  → 11أ\n"
            "  'تسعة ألف'      → 9أ\n"
            "  'عشرة باء'      → 10ب\n"
            f"Phrase: {spoken}\n"
            "Reply:"
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=5,
        ),
    )
    result = response.text.strip().split()[0] if response.text.strip() else ""
    print(f"  [Gemini Class] '{ar(result)}' ← '{ar(spoken)}'")
    return result


# ── Step 2: class verification ────────────────────────────────────────────────

def verify_class(candidate: dict) -> bool:
    """
    Ask the student to STATE their class without revealing the expected answer.
    normalize_class() first, Gemini fallback if section is missing.

    Retry messages are case-specific:
      no audio  → "مَا سَمَعْتِجْ زِيْنْ..."
      heard but wrong → "رَقَمْ اَلصَّفّْ مَا طَابَقْ..."

    3 attempts total.
    """
    for attempt in range(1, CLASS_ATTEMPTS + 1):
        print(f"\n  [Step 2] Attempt {attempt}/{CLASS_ATTEMPTS}")

        # First attempt: initial open question
        # Attempts 2+: retry message already spoken at end of previous attempt
        if attempt == 1:
            speak("وِشْ صَفِّجْ؟")

        answer = listen()

        # ── Case 1: no audio ──────────────────────────────────────────────
        if not answer:
            if attempt < CLASS_ATTEMPTS:
                speak("مَا سَمَعْتِجْ زِيْنْ، عَادْ قُولِي لِي رَقَمْ صَفِّجْ وَشُعْبَتِجْ.")
            continue

        # ── Case 2: audio received → parse and compare ────────────────────
        norm_given  = normalize_class(answer)
        norm_stored = normalize_class(candidate['class_code'])

        # Gemini fallback if deterministic parse found no Arabic section letter
        if norm_given and not re.search(r'[\u0600-\u06FF]', norm_given):
            print(f"  [Step 2] No section detected — Gemini fallback")
            norm_given = gemini_normalize_class(answer)

        if norm_given and norm_given == norm_stored:
            print(f"  [Step 2] ✓ Class match: "
                  f"'{ar(norm_given)}' == '{ar(norm_stored)}'")
            return True

        print(f"  [Step 2] ✗ Mismatch: "
              f"'{ar(norm_given)}' ≠ '{ar(norm_stored)}'")

        if attempt < CLASS_ATTEMPTS:
            speak("رَقَمْ اَلصَّفّْ مَا طَابَقْ. عَادْ قُولِي رَقَمْ اَلصَّفّْ وَٱلشُّعْبَةْ.")

    speak(f"ما أَگْدَرْ أكمل. الصَّفّْ ما طَابَقّْ السِّجِل. "
          f"بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
    return False


# ── Step 3: final identity confirmation ───────────────────────────────────────

def final_confirmation(candidate: dict) -> bool:
    """
    Amal reads back BOTH full name and class together.
    Student must explicitly confirm before health profile is accessed.
    Only True on explicit YES. No audio / UNCLEAR / NO → False.
    """
    speak(
        f"إِنْتِي {candidate['full_name']}، "
        f"مِنْ الصَّفّْ {candidate['class_lib']}، "
        f"صَحّْ؟"
    )

    answer = listen()

    if not answer:
        print("  [Step 3] ✗ No audio — confirmation required")
        speak(f"ما سمعتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
        return False

    result = is_affirmative(answer)

    if result is True:
        print(f"  [Step 3] ✓ Confirmed: "
              f"{ar(candidate['full_name'])} | {candidate['class_code']}")
        return True

    if result is False:
        print("  [Step 3] ✗ Student denied identity")
        speak(f"زِيْنّْ. بتواصل مع الممرضة {NURSE_NAME} "
              f"لتصحيح السِّجِل، هي تجي تساعدِجْ.")
        return False

    print(f"  [Step 3] ✗ Unclear: '{ar(answer)}'")
    speak(f"ما فهمتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
    return False


# ── Full 3-step identification loop ───────────────────────────────────────────

def identify_loop() -> dict | None:
    """
    Step 1 — Full name (up to MAX_ATTEMPTS)
      STT → clean prefix → single-token check → phonetic fuzzy → candidate
      Both avg >= FUZZY_THRESHOLD AND min >= MIN_TOKEN_SCORE required

    Step 2 — Class verification (up to CLASS_ATTEMPTS)
      Open question → STT → normalize_class() → Gemini fallback → exact match

    Step 3 — Final confirmation
      Amal reads full name + class → student confirms → YES only
    """
    candidate = None

    # ── Step 1: name ──────────────────────────────────────────────────────
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  [Step 1] Attempt {attempt}/{MAX_ATTEMPTS}")

        speak("وِشْ إِسْمِجْ الكامل؟" if attempt == 1
              else "ما فهمتِجْ زين، عاد قولي إِسْمِجْ الكامل.")

        stt_text = listen()
        if not stt_text:
            continue

        full_name = clean_name_from_stt(stt_text)
        if not full_name:
            continue

        # Single token = first name only → ask "X ءايش؟" and re-listen
        if len(full_name.split()) == 1:
            print(f"  [Step 1] Single token '{ar(full_name)}' — asking for full name")
            speak(f"{full_name} ءَايِشْ؟")
            stt_text2 = listen()
            if stt_text2:
                full_name2 = clean_name_from_stt(stt_text2)
                if full_name2:
                    full_name = full_name2

            # Still single token or no response → retry, never fuzzy on first name alone
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

    # ── Step 2: class verification ────────────────────────────────────────
    print(f"\n  [Step 2] Verifying class for {ar(candidate['full_name'])}")
    if not verify_class(candidate):
        return None

    # ── Step 3: final confirmation ────────────────────────────────────────
    print(f"\n  [Step 3] Final confirmation for {ar(candidate['full_name'])}")
    if not final_confirmation(candidate):
        return None

    return candidate


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== test_identify.py — 3-step patient identification ===\n")

    init_db()

    speak("هالا وغالا! أنا أَمَلْ، مساعدتِجْ الذكية في العيادة.")

    patient = identify_loop()

    if patient:
        print(f"\n  [DB] ✓ {ar(patient['full_name'])} | "
              f"{patient['class_code']} | {patient['urgency_level']}")
        print("\n--- Patient context (LLM injection) ---")
        for line in patient_context(patient["id"]).split("\n"):
            print(f"  {ar(line)}" if line.strip() else "")
        print("---------------------------------------\n")
        speak(
            f"زِيْنّْ! وِشْ فِيجْ اَلْيُومْ "
            f"{patient['full_name'].split()[0]}؟"
        )
    else:
        print("\n  [DB] Identification failed.")


if __name__ == "__main__":
    main()
