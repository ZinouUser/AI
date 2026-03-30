# debug_audio.py
import os, subprocess
from elevenlabs import ElevenLabs

API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
MODEL    = "eleven_v3"

print(f"API_KEY  : {'✓ défini' if API_KEY  else '✗ VIDE'}")
print(f"VOICE_ID : {'✓ ' + VOICE_ID if VOICE_ID else '✗ VIDE'}")

# 1. Test TTS minimal
print("\n[1] Appel TTS...")
try:
    client = ElevenLabs(api_key=API_KEY)
    audio = b"".join(client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text="هلا",
        model_id=MODEL,
        language_code="ar",
    ))
    print(f"    Bytes reçus : {len(audio)}")
except Exception as e:
    print(f"    ERREUR TTS : {e}")
    exit(1)

# 2. Sauvegarde
with open("debug_test.mp3", "wb") as f:
    f.write(audio)
print("    Fichier : debug_test.mp3")

# 3. Test lecture ffplay
print("\n[2] Lecture ffplay...")
result = subprocess.run(
    ["ffplay", "-nodisp", "-autoexit", "debug_test.mp3"],
    capture_output=True, text=True
)
print(f"    ffplay code retour : {result.returncode}")
if result.stderr:
    print(f"    ffplay stderr : {result.stderr[:300]}")

# 4. Test micro (enregistrement 3 secondes)
print("\n[3] Test microphone (3 secondes)...")
result2 = subprocess.run(
    ["ffmpeg", "-y", "-f", "alsa", "-i", "default",
     "-t", "3", "debug_mic.wav"],
    capture_output=True, text=True
)
print(f"    ffmpeg code retour : {result2.returncode}")
if result2.returncode != 0:
    print(f"    stderr : {result2.stderr[-300:]}")
else:
    size = os.path.getsize("debug_mic.wav")
    print(f"    Fichier micro : debug_mic.wav ({size} bytes)")
