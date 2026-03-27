"""
Test STT — valide la reconnaissance vocale arabe Qatar.
Usage : python test_stt.py  → parle en arabe après "En écoute..."
"""
import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

cfg = speechsdk.SpeechConfig(
    subscription=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION", "qatarcentral"),
)
cfg.speech_recognition_language = "ar-QA"

recognizer = speechsdk.SpeechRecognizer(speech_config=cfg)
print("🎤 En écoute... (parle en arabe)")

result = recognizer.recognize_once_async().get()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print(f"✅ Reconnu : {result.text}")
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("⚠️  Rien reconnu.")
else:
    print(f"❌ Erreur : {result.reason}")
    if hasattr(result, "cancellation_details"):
        print(result.cancellation_details.error_details)

