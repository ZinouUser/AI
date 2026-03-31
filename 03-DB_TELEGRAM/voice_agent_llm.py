"""
voice_agent_llm.py
──────────────────
Amal Voice Agent — Azure STT + Gemini Flash LLM + ElevenLabs TTS
Phase 03 — DB + RAG + Identification

Usage:
    python voice_agent_llm.py --mode sim    # PC simulation (default)
    python voice_agent_llm.py --mode real   # physical TonyPi robot
"""

import os
import sys
import re
import sqlite3
import argparse
import warnings
import tempfile
import subprocess
from difflib import SequenceMatcher

import arabic_reshaper
from bidi.algorithm import get_display
import azure.cognitiveservices.speech as speechsdk
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
from dotenv import load_dotenv

from gemini_brain import GeminiBrain
from qatari_dialect import QatariDialect
from tashkil_display import TashkilDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_patients import init_db, patient_context, log_visit, DB_PATH
from db_protocols import init_protocols, protocol_context

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

SPEECH_KEY          = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION       = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
STT_LANGUAGE        = "ar-QA"
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
MAX_TURNS       = 8

eleven  = ElevenLabs(api_key=ELEVENLABS_KEY)
dialect = QatariDialect()
display = TashkilDisplay()
display.enabled = True

EXIT_KEYWORDS = ["وداعاً", "باي", "إنهاء", "أوقف", "خروج", "مع السلامة", "انتهى"]


# ── Arabic display helpers ─────────────────────────────────────────────────────

def ar(text: str) -> str:
    """Inline Arabic in English f-string → reshape + bidi."""
    return get_display(arabic_reshaper.reshape(text))


# ── STT ───────────────────────────────────────────────────────────────────────

def build_stt_config() -> speechsdk.SpeechConfig:
    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY missing in .env")
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = STT_LANGUAGE
    # Give student more time to start speaking and to finish
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "8000"
    )
    cfg.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "2000"
    )
    return cfg


def _is_arabic(text: str) -> bool:
    """Return True if at least 30% of non-space chars are Arabic."""
    chars = text.replace(' ', '')
    if not chars:
        return False
    return len(re.findall(r'[\u0600-\u06FF]', chars)) / len(chars) >= 0.30


def listen(cfg: speechsdk.SpeechConfig) -> str:
    recognizer = speechsdk.SpeechRecognizer(speech_config=cfg)
    print("\n  Listening...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Discard non-Arabic noise mis-recognized as English
        if not _is_arabic(result.text):
            print(f"  [STT] Non-Arabic discarded: '{result.text}'")
            return ""
        display.print_arabic("  User : ", result.text)
        return result.text
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

    # 1. Afficher le tashkil brut de Gemini (vérification)
    display.show_tashkil("Gemini tashkil", text)

    # 2. Pipeline dialectal pour l'affichage uniquement
    display_text = dialect.process(text)
    display.print_arabic("  Amal : ", display_text)

    # 3. Texte pour TTS = Gemini RAW avec tashkil complet
    tts_text = text.replace("\u06AF", "\u0642")
    tts_text = tts_text.rstrip() + " ."

    print(f"  TTS reçoit : {tts_text}")

    # 4. Appel ElevenLabs
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
    # os.unlink(mp3_path)

    # ── OPTION B : WSL (actif en développement PC) ────────────────────────
    wav_path = mp3_path.replace(".mp3", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-af", "apad=pad_dur=0.5", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.run(["aplay", "-q", wav_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(mp3_path)
    os.unlink(wav_path)

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
    text = re.sub(r'[^\u0600-\u06FF\s0-9a-zA-Z]', ' ', text)
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
        display.print_arabic("Fuzzy ✓", f"{best['full_name']}  score={best_score:.2f}")
        return best
    print(f"  [Fuzzy] ✗ No match (best={best_score:.2f})")
    return None


_NAME_PREFIXES = [
    "أنا اسمي","أنا إسمي","اسمي","إسمي","أنا",
    "اسمه","إسمه","اسمها","إسمها",
]


def clean_name_from_stt(stt_text: str) -> str:
    text = stt_text.strip().rstrip(".")
    for prefix in sorted(_NAME_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    display.print_arabic("Name", text)
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
    for attempt in range(1, CLASS_ATTEMPTS + 1):
        print(f"\n  [Step 2] Attempt {attempt}/{CLASS_ATTEMPTS}")
        if attempt == 1:
            speak("وِشْ صَفِّجْ؟")
        answer = listen(cfg)
        if not answer:
            if attempt < CLASS_ATTEMPTS:
                speak("مَا سَمَعْتِجْ زِيْنْ، عَادْ قُولِي لِي رَقَمْ صَفِّجْ وَشُعْبَتِجْ.")
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
    answer = listen(cfg)
    if not answer:
        speak(f"ما سمعتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
        return False
    result = is_affirmative(answer)
    if result is True:
        print(f"  [Step 3] ✓ {ar(candidate['full_name'])} confirmed")
        return True
    if result is False:
        speak(f"زِيْنّْ. بتواصل مع الممرضة {NURSE_NAME} لتصحيح السِّجِل.")
        return False
    speak(f"ما فهمتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
    return False


def identify_loop(cfg: speechsdk.SpeechConfig) -> dict | None:
    candidate = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  [Step 1] Attempt {attempt}/{MAX_ATTEMPTS}")
        speak("وِشْ إِسْمِجْ الكامل؟" if attempt == 1
              else "ما فهمتِجْ زين، عاد قولي إِسْمِجْ الكامل.")
        stt_text = listen(cfg)
        if not stt_text:
            continue
        full_name = clean_name_from_stt(stt_text)
        if not full_name:
            continue
        if len(full_name.split()) == 1:
            print(f"  [Step 1] Single token '{ar(full_name)}' — asking for full name")
            speak(f"{full_name} ءَايِشْ؟")
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


# ── Consultation loop (replaces run_loop) ─────────────────────────────────────

def consultation_loop(cfg: speechsdk.SpeechConfig, brain: GeminiBrain,
                      patient: dict):
    """
    Medical consultation after identification.
    Each turn: STT → RAG (db_protocols) → brain.set_context() → think() → TTS
    Visit logged in SQLite at end.
    """
    first_name   = patient['full_name'].split()[0]
    symptoms_log = []

    # Urgency-aware greeting
    if patient.get('urgency_level') == 'urgent':
        speak(f"زِيْنّْ {first_name}! عندِجْ ملف خاص عندنا. وِشْ فِيجْ اليوم؟")
    else:
        speak(f"زِيْنّْ {first_name}! وِشْ فِيجْ اليوم؟")

    user_text = listen(cfg)
    if not user_text:
        speak(f"ما سمعتِجْ. بتواصل مع الممرضة {NURSE_NAME}، هي تجي تساعدِجْ.")
        return

    symptoms_log.append(user_text)
    turn = 0

    while turn < MAX_TURNS:
        turn += 1

        if any(kw in user_text for kw in EXIT_KEYWORDS):
            break

        # RAG: fetch relevant protocols for current symptoms
        protos = protocol_context(user_text, n_results=2)
        print(f"  [RAG] Protocols updated for turn {turn}")

        # Inject patient + protocol context into brain
        brain.set_context(
            patient_ctx=patient_context(patient["id"]),
            protocol_ctx=protos,
        )

        reply = brain.think(user_text)
        speak(reply)

        user_text = listen(cfg)
        if not user_text:
            continue
        if any(kw in user_text for kw in EXIT_KEYWORDS):
            break
        symptoms_log.append(user_text)

    speak(f"تمام {first_name}! اللَّهْ يَشْفِيجْ. "
          f"لو احتجتِ شي، الممرضة {NURSE_NAME} موجودة.")

    # Log visit in SQLite
    log_visit(
        patient_id=patient["id"],
        symptoms_ar=" | ".join(symptoms_log),
        action_ar=f"تقييم بواسطة {AI_NAME} — {turn} دورة",
        protocol_id="auto-rag",
        notified=False,
    )
    print(f"  [DB] ✓ Visit logged — {ar(patient['full_name'])} | turns={turn}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amal voice agent")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    args = parser.parse_args()

    display.init_log()
    print(f"\n=== Amal | STT=Azure ar-QA | LLM=Gemini Flash | "
          f"TTS=ElevenLabs v3 | mode={args.mode} ===\n")
    print(f"  [{'REAL' if args.mode == 'real' else 'SIM'} MODE]\n")

    # Init databases
    init_db()
    init_protocols()

    cfg   = build_stt_config()
    brain = GeminiBrain()

    speak("هَلَا وَاللَّهْ! أَنَا أَمَلْ، مُسَاعِدَتِجْ الذَّكِيَّةْ فِي الْعِيَادَةْ.")

    # Step 1-2-3: identify patient
    patient = identify_loop(cfg)
    if not patient:
        print("\n  [Agent] Identification failed — session ended.")
        return

    print(f"\n  [Agent] ✓ {ar(patient['full_name'])} | "
          f"{patient['class_code']} | {patient['urgency_level']}\n")

    # Consultation with RAG + patient context
    consultation_loop(cfg, brain, patient)
    print("\n  [Agent] Session complete.\n")


if __name__ == "__main__":
    main()
