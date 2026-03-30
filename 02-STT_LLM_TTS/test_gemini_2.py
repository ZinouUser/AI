"""
Acceptance Tests — GeminiBrain Part 2/2  (T5 to T7)
No microphone or audio required — text only.

  T5 — Conversational memory (first name retained across turns)
  T6 — Local fallback returned on simulated TimeoutError
  T7 — reset() clears conversation memory

Usage: python test_gemini_2.py
Run at least 60 seconds after test_gemini_1.py (Gemini free tier: 15 RPM).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import concurrent.futures
from unittest.mock import patch
import arabic_reshaper
from bidi.algorithm import get_display
from gemini_brain import GeminiBrain, FALLBACK_RESPONSES

TASHKIL_CHARS = set("ًٌٍَُِّْٰٖٗ")


def print_arabic(label: str, text: str):
    """Print Arabic text correctly oriented in the terminal (RTL fix)."""
    print(f"{label}{get_display(arabic_reshaper.reshape(text))}")


def strip_tashkil(text: str) -> str:
    """Remove Arabic diacritics so string comparisons work regardless of tashkil."""
    return "".join(c for c in text if c not in TASHKIL_CHARS)


def extract_speech(reply: str) -> str:
    """
    Extract the 'speech' field from Gemini's JSON response.
    Handles: plain JSON, ```json fences, truncated JSON.
    Returns raw text if all parsing attempts fail.
    """
    clean = reply.strip()

    if "```" in clean:
        start = clean.find("```") + 3
        end   = clean.rfind("```")
        if end > start:
            clean = clean[start:end]
        if clean.lstrip().startswith("json"):
            clean = clean.lstrip()[4:]
        clean = clean.strip()

    try:
        data = json.loads(clean)
        return data.get("speech", reply)
    except json.JSONDecodeError:
        pass

    match = re.search(r'"speech"\s*:\s*"([^"]+)"', clean, re.DOTALL)
    if match:
        return match.group(1)

    return reply


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_memory():
    """T5 — First name is retained across conversation turns. (2 API calls)"""
    brain = GeminiBrain()
    brain.think("اسمي فاطمة")
    reply = brain.think("وش اسمي؟")
    speech = extract_speech(reply)
    print_arabic("  speech : ", speech)
    assert "فاطمة" in strip_tashkil(speech), \
        f"First name not retained in memory: {strip_tashkil(speech)}"


def test_fallback_on_timeout():
    """T6 — Local fallback is returned when Gemini raises TimeoutError. (0 API calls)"""
    brain = GeminiBrain()
    with patch.object(brain, "_call_gemini",
                      side_effect=concurrent.futures.TimeoutError()):
        reply = brain.think("أي رسالة")
    print_arabic("  fallback : ", reply)
    assert reply in FALLBACK_RESPONSES, f"Unexpected fallback response: {reply}"


def test_reset_clears_memory():
    """T7 — reset() clears the conversation history. (2 API calls)"""
    brain = GeminiBrain()
    brain.think("اسمي سارة")
    brain.reset()
    reply = brain.think("وش اسمي؟")
    speech = extract_speech(reply)
    print_arabic("  speech after reset : ", speech)
    assert "سارة" not in strip_tashkil(speech), \
        f"Memory not cleared after reset: {strip_tashkil(speech)}"


# ── Runner ─────────────────────────────────────────────────────────────────────

TESTS = [
    ("T5 — Conversation memory",  test_memory),
    ("T6 — Fallback on timeout",  test_fallback_on_timeout),
    ("T7 — Reset memory",         test_reset_clears_memory),
]

if __name__ == "__main__":
    print("=" * 55)
    print("  Acceptance Tests — GeminiBrain  [Part 2/2]")
    print("=" * 55)

    passed, failed = 0, 0
    for name, fn in TESTS:
        print(f"\n--- {name} ---")
        try:
            fn()
            print("  OK")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL : {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR : {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Result : {passed}/{len(TESTS)} tests passed")
    if failed == 0:
        print("  All 7 tests passed — GeminiBrain is fully operational")
    else:
        print(f"  {failed} test(s) failed — see details above")
    print("=" * 55)
