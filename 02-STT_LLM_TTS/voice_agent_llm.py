"""
Amal Voice Agent — Azure STT + Gemini Flash LLM + ElevenLabs TTS
S1 — Full loop with LLM brain

Usage:
    python voice_agent_llm.py --mode sim    # PC simulation (default)
    python voice_agent_llm.py --mode real   # physical TonyPi robot (S2)
"""

import os
import argparse
import warnings
import tempfile
import subprocess
import azure.cognitiveservices.speech as speechsdk
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

from gemini_brain import GeminiBrain
from qatari_dialect import QatariDialect
from tashkil_display import TashkilDisplay

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

SPEECH_KEY          = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION       = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
STT_LANGUAGE        = "ar-QA"
ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL    = "eleven_v3"

eleven  = ElevenLabs(api_key=ELEVENLABS_KEY)
dialect = QatariDialect()
display = TashkilDisplay()
display.enabled = True   # False = désactive tout l'affichage tashkil

EXIT_KEYWORDS = ["وداعاً", "باي", "إنهاء", "أوقف", "خروج", "مع السلامة", "انتهى"]


# ── STT ───────────────────────────────────────────────────────────────────────

def build_stt_config() -> speechsdk.SpeechConfig:
    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY missing in .env")
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = STT_LANGUAGE
    return cfg


def listen(cfg: speechsdk.SpeechConfig) -> str:
    recognizer = speechsdk.SpeechRecognizer(speech_config=cfg)
    print("\n  Listening...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
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
    tts_text = tts_text.rstrip() + " ."        # ← fix coupure dernier mot

    # ── VÉRIFICATION : ce que reçoit ElevenLabs ──
    print(f"  TTS reçoit : {tts_text}")
    # ─────────────────────────────────────────────

    # 4. Appel ElevenLabs
    audio_generator = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=tts_text,
        model_id=ELEVENLABS_MODEL,
        language_code="ar",
    )
    audio_bytes = b"".join(audio_generator)

    # Sauvegarde pour inspection
    # with open("/tmp/last_tts.mp3", "wb") as f:
    #     f.write(audio_bytes)
    # print(f"  [DEBUG] audio bytes : {len(audio_bytes)} → /tmp/last_tts.mp3")


    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        mp3_path = tmp.name

    # ── OPTION A : Ubuntu natif / TonyPi Pro (décommenter pour le robot) ──
    # subprocess.run(
    #     ["mpg123", "-q", mp3_path],
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL
    # )
    # os.unlink(mp3_path)

    # ── OPTION B : WSL (actif en développement PC) ────────────────────────
    wav_path = mp3_path.replace(".mp3", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path,
        "-af", "apad=pad_dur=0.5",
        wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["aplay", "-q", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    os.unlink(mp3_path)
    os.unlink(wav_path)

    return True


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(cfg: speechsdk.SpeechConfig, brain: GeminiBrain):
    speak("هَلَا وَاللَّهْ! أَنَا أَمَلْ. تِكَلَّمِي وَأَنَا أَجَاوِبِجْ.")
    while True:
        user_text = listen(cfg)
        if not user_text:
            continue
        if any(kw in user_text for kw in EXIT_KEYWORDS):
            speak("يَلَّا، مَعَ السَّلَامَةْ! اَللَّهْ يَحْفِظِجْ.")
            break
        reply = brain.think(user_text)
        speak(reply)


def main():
    parser = argparse.ArgumentParser(description="Amal voice agent — STT + Gemini LLM + TTS")
    parser.add_argument(
        "--mode", choices=["sim", "real"], default="sim",
        help="sim = PC simulation (default) | real = physical TonyPi robot"
    )
    args = parser.parse_args()

    display.init_log()
    print(f"\n=== Amal | STT=Azure ar-QA | LLM=Gemini | "
          f"TTS=ElevenLabs v3 | mode={args.mode} ===\n")
    if args.mode == "real":
        print("  [REAL MODE] TonyPi Pro robot active\n")
    else:
        print("  [SIM MODE] PC simulation\n")

    cfg   = build_stt_config()
    brain = GeminiBrain()
    run_loop(cfg, brain)


if __name__ == "__main__":
    main()
