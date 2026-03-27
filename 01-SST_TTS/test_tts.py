"""
Test TTS — valide la clé ElevenLabs et la voix (sans micro).
Usage : python test_tts.py
"""
import os
import time
import warnings
import tempfile
import subprocess
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display
from elevenlabs import VoiceSettings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()

ELEVENLABS_KEY      = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL    = "eleven_v3"

eleven = ElevenLabs(api_key=ELEVENLABS_KEY)

_reshaper = ArabicReshaper(configuration={'keep_harakat': True})

def ar(text: str) -> str:
    return get_display(_reshaper.reshape(text))

SENTENCES = [
    "َحَسَبِ البْرُتُوكُولِ المُعْتَمَدْ",
    "هَاذَا الِّـي لَازِمْ أَسَوِّيهِ الحِينْ",
    "أَوَّلْ شَيْ، أَخَبِّرْ إِدَارَةِ المَدْرَسَهْ بِاللِّي صَارْ",
    "الثَّانِي، أَرْسِلْ رِسَالَةْ لأَهِلْ سَارَهْ وأَطَمِّنْهُم",
    "الثَّالث، أَسَجِّلِ الحَادِثَةْ فِي مَلَفّْ سَارَ تِلْقَائِيًّا"
]

for sentence in SENTENCES:
    print(f"🔊 Synthèse en cours : {ar(sentence)}")

    audio_generator = eleven.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=sentence,
        model_id=ELEVENLABS_MODEL,
        language_code="ar",
        # voice_settings=VoiceSettings(
        #     speed=1.2,  # 0.7 = lent, 1.0 = normal, 1.2 = rapide
        # ),
    )
    audio_bytes = b"".join(audio_generator)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-af", "atempo=1.3", tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    finally:
        os.unlink(tmp_path)
        #print(f"📁 Fichier sauvegardé : {tmp_path}")


    #time.sleep(0.3)

print("✅ Lecture terminée !")  
