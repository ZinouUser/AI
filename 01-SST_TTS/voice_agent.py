"""
Agent Vocal Arabe - Azure STT + ElevenLabs TTS
Robot TonyPi Pro / Raspberry Pi 5
"""

import os
import argparse
import warnings
import tempfile
import subprocess
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import arabic_reshaper
from bidi.algorithm import get_display

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()

SPEECH_KEY          = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION       = os.getenv("AZURE_SPEECH_REGION", "qatarcentral")
STT_LANGUAGE        = "ar-QA"
ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL    = "eleven_v3"

eleven = ElevenLabs(api_key=ELEVENLABS_KEY)


def print_arabic(label: str, text: str):
    print(f"{label}{get_display(arabic_reshaper.reshape(text))}")


def build_stt_config() -> speechsdk.SpeechConfig:
    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY manquant dans .env")
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = STT_LANGUAGE
    return cfg


def listen(cfg: speechsdk.SpeechConfig) -> str:
    recognizer = speechsdk.SpeechRecognizer(speech_config=cfg)
    print("\n🎤 En écoute...")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print_arabic("  👤 Vous : ", result.text)
        return result.text
    if result.reason == speechsdk.ResultReason.NoMatch:
        print("  ⚠️  Rien reconnu, réessaie.")
        return ""
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"  ❌ STT annulé : {details.reason} — {details.error_details}")
        return ""
    return ""


def speak(text: str) -> bool:
    if not ELEVENLABS_KEY or not ELEVENLABS_VOICE_ID:
        raise EnvironmentError("ELEVENLABS_API_KEY ou ELEVENLABS_VOICE_ID manquant dans .env")

    print_arabic("  🔊 أمل : ", text)

    audio_generator = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id=ELEVENLABS_MODEL,
    )
    audio_bytes = b"".join(audio_generator)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        subprocess.run(["mpg123", "-q", tmp_path], check=True)
    finally:
        os.unlink(tmp_path)

    return True


def get_response(question: str) -> str:
    if any(kw in question for kw in ["اسمك", "من أنت", "وش أنت"]):
        return "والله، أنا أمل! روبوت ذكي من توني باي. كيف أقدر أساعدك؟"
    if any(kw in question for kw in ["شلونك", "كيفك", "كيف حالك", "عساك بخير"]):
        return "الحمدلله، زين وبخير! وأنت شلونك؟"
    if any(kw in question for kw in ["الوقت", "الساعة", "كم الساعة"]):
        now = datetime.now().strftime("%H:%M")
        return f"الساعة الحين {now}. عاد وش تبي؟"
    if any(kw in question for kw in ["عاصمة", "قطر", "الدوحة"]):
        return "والله عاصمة قطر الدوحة، خوش بلد!"
    if any(kw in question for kw in ["شكراً", "مشكور", "يسلموا"]):
        return "هلا والله! ماشي، أي خدمة."
    if any(kw in question for kw in ["مرحبا", "هلا", "السلام"]):
        return "هلا هلا! يا هلا بيك. وش تبي؟"
    return "والله ما فهمت زين. عاد قولها مرة ثانية؟"


def mode_user_asks(cfg: speechsdk.SpeechConfig):
    speak("هلا والله! أنا أمل. تكلّم وأنا أجاوبك.")
    while True:
        question = listen(cfg)
        if not question:
            continue
        if any(kw in question for kw in ["وداعاً", "باي", "إنهاء", "أوقف", "خروج"]):
            speak("يلا، مع السلامة!")
            break
        speak(get_response(question))


ROBOT_QUESTIONS = [
    "هلا! وش اسمك؟",
    "من وين أنت؟",
    "وش هوايتك المفضلة؟",
]


def mode_robot_asks(cfg: speechsdk.SpeechConfig):
    speak("هلا! أنا أمل. بسألك شوية أسئلة، زين؟")
    for question in ROBOT_QUESTIONS:
        speak(question)
        answer = listen(cfg)
        if answer:
            speak(f"والله خوش! قلت: {answer}")
        else:
            speak("ما سمعتك، عاد نكمل.")
    speak("شكراً لك! كان نقاش زين. يلا مع السلامة.")


def main():
    parser = argparse.ArgumentParser(description="Agent vocal qatari - Azure STT + ElevenLabs TTS")
    parser.add_argument("--mode", choices=["user", "robot"], default="user")
    args = parser.parse_args()
    cfg = build_stt_config()
    print(f"\n=== Agent Vocal | STT=Azure ar-QA | TTS=ElevenLabs | mode={args.mode} ===\n")
    if args.mode == "user":
        mode_user_asks(cfg)
    else:
        mode_robot_asks(cfg)


if __name__ == "__main__":
    main()
