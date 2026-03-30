"""
Integration Test — Gemini LLM + TTS (no mic)
Target: LLM + TTS latency < 15s per turn.

After run: open tashkil_debug.log in VS Code EDITOR (not cat) for RTL + tashkil display.

Usage: python test_full_loop.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import warnings
import tempfile
import subprocess
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from gemini_brain import GeminiBrain
from qatari_dialect import QatariDialect
from tashkil_display import TashkilDisplay

display = TashkilDisplay()
display.enabled = False

warnings.filterwarnings("ignore")
load_dotenv()

ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL    = "eleven_v3"

eleven  = ElevenLabs(api_key=ELEVENLABS_KEY)
brain   = GeminiBrain()
dialect = QatariDialect()
display = TashkilDisplay()


# ── TTS ───────────────────────────────────────────────────────────────────────

def speak_and_measure(raw_text: str) -> float:
    """Apply dialect pipeline, send to TTS, play audio. Returns TTS latency."""

    # Qatari dialect pipeline
    text = dialect.process(raw_text)
    text = text.rstrip() + " ."                  # fix coupure dernier mot
    display.print_arabic("  Amal TTS : ", text)

    t0 = time.time()
    audio_generator = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id=ELEVENLABS_MODEL,
        language_code="ar",
    )
    audio_bytes = b"".join(audio_generator)
    tts_latency = time.time() - t0

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        mp3_path = f.name

    try:
        # ── OPTION A : Ubuntu natif / TonyPi Pro (décommenter pour le robot) ──
        # subprocess.run(
        #     ["mpg123", "-q", mp3_path],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL
        # )

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
        os.unlink(wav_path)
        # ──────────────────────────────────────────────────────────────────────

    finally:
        os.unlink(mp3_path)

    return tts_latency


# ── Test runner ───────────────────────────────────────────────────────────────

def run_turn(label: str, user_text: str, max_total_s: float = 15.0) -> float:
    print(f"\n--- {label} ---")
    display.print_arabic("  Input  : ", user_text)

    # LLM
    t0        = time.time()
    llm_reply = brain.think(user_text)
    t_llm     = time.time() - t0

    # Tashkil verification: count in terminal + raw text in log file
    display.show_tashkil("Gemini tashkil", llm_reply)

    # Normal display
    display.print_arabic("  Gemini : ", llm_reply)
    print(f"  Gemini : {t_llm:.2f}s")

    # TTS
    t0          = time.time()
    tts_latency = speak_and_measure(llm_reply)
    t_tts       = time.time() - t0

    total = t_llm + t_tts
    print(f"  TTS    : {tts_latency:.2f}s")
    print(f"  Total  : {total:.2f}s  (cible < {max_total_s}s)")

    assert total < max_total_s, (
        f"Latency {total:.2f}s exceeds target {max_total_s}s"
    )
    return total


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("Tour 1 — Salutation",         "هلا، شلونك؟",                  15.0),
    ("Tour 2 — Identité",           "وش اسمك؟",                     15.0),
    ("Tour 3 — Symptôme abdominal", "عندي ألم في بطني من الصباح",   15.0),
    ("Tour 4 — Allergie",           "عندي حساسية من الحليب",         15.0),
    ("Tour 5 — Mémoire",            "وش قلت قبل عن الحساسية؟",      20.0),
]


if __name__ == "__main__":
    display.init_log()

    print("=" * 58)
    print("  Integration Test — Gemini LLM + TTS (no mic)")
    print("  Tashkil log → open tashkil_debug.log in VS Code EDITOR")
    print("=" * 58)

    results = []
    for label, text, max_s in SCENARIOS:
        try:
            latency = run_turn(label, text, max_s)
            results.append((label, latency, True))
            print("  OK")
        except AssertionError as e:
            print(f"  FAIL : {e}")
            results.append((label, None, False))
        except Exception as e:
            print(f"  ERROR : {type(e).__name__}: {e}")
            results.append((label, None, False))

    print(f"\n{'='*58}")
    print("  Latency summary :")
    passed = sum(1 for _, _, ok in results if ok)
    for label, lat, ok in results:
        status  = "OK  " if ok else "FAIL"
        lat_str = f"{lat:.2f}s" if lat is not None else "N/A"
        print(f"  {status}  {label:<40} {lat_str}")

    print(f"\n  {passed}/{len(results)} turns passed")
    if passed == len(results):
        print("  LLM+TTS operational — ready for STT mic integration")
    print(f"  Tashkil details → tashkil_debug.log (open in VS Code editor)")
    print("=" * 58)
