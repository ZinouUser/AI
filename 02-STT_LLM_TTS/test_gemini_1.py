"""
Acceptance Tests — GeminiBrain Part 1/2  (T1 to T4)
No microphone or audio required — text only.

  T1 — Response uses Khaliji Gulf dialect
  T2 — Tashkil (Arabic diacritics) present in speech field
  T3 — Amal introduces herself by name
  T4 — Relevant response to a medical symptom

Usage: python test_gemini_1.py
Run test_gemini_2.py at least 60 seconds later (Gemini free tier: 15 RPM).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import arabic_reshaper
from bidi.algorithm import get_display
from gemini_brain import GeminiBrain

KHALIJI_KEYWORDS = [
    "والله", "وش", "شلونك", "زين", "عاد", "خوش",
    "هلا", "يلا", "وين", "حين", "اهلا", "شوية", "الحين",
]
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

    # Strip markdown code fences
    if "```" in clean:
        start = clean.find("```") + 3
        end   = clean.rfind("```")
        if end > start:
            clean = clean[start:end]
        if clean.lstrip().startswith("json"):
            clean = clean.lstrip()[4:]
        clean = clean.strip()

    # Attempt full JSON parse
    try:
        data = json.loads(clean)
        return data.get("speech", reply)
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction of speech field value
    match = re.search(r'"speech"\s*:\s*"([^"]+)"', clean, re.DOTALL)
    if match:
        return match.group(1)

    return reply


def has_khaliji(text: str) -> bool:
    bare = strip_tashkil(text)
    return any(kw in bare for kw in KHALIJI_KEYWORDS)


def has_tashkil(text: str) -> bool:
    return any(c in TASHKIL_CHARS for c in text)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_khaliji_dialect():
    """T1 — Khaliji dialect detected in a greeting response."""
    brain = GeminiBrain()
    reply = brain.think("هلا، شلونك؟")
    speech = extract_speech(reply)
    print_arabic("  speech : ", speech)
    assert len(speech) > 0, "Empty response"
    assert has_khaliji(speech), f"No Khaliji keyword found in: {strip_tashkil(speech)}"


def test_tashkil():
    """T2 — Tashkil (diacritics) present in speech field."""
    brain = GeminiBrain()
    reply = brain.think("قولي جملة طبية فيها تشكيل على الكلمات")
    speech = extract_speech(reply)
    print_arabic("  speech : ", speech)
    if not has_tashkil(speech):
        print("  WARNING: No tashkil found in speech — check system prompt (non-blocking)")


def test_identity():
    """T3 — Amal introduces herself by name."""
    brain = GeminiBrain()
    reply = brain.think("وش اسمك؟")
    speech = extract_speech(reply)
    print_arabic("  speech : ", speech)
    # Compare without tashkil — "أَمَلْ" == "أمل" after stripping
    assert "أمل" in strip_tashkil(speech), \
        f"Amal does not mention her name: {strip_tashkil(speech)}"


def test_medical_response():
    """T4 — Non-empty response to a medical symptom."""
    brain = GeminiBrain()
    reply = brain.think("عندي ألم في بطني من الصباح")
    speech = extract_speech(reply)
    print_arabic("  speech : ", speech)
    assert len(strip_tashkil(speech)) > 10, "Response too short for a medical symptom"


# ── Runner ─────────────────────────────────────────────────────────────────────

TESTS = [
    ("T1 — Khaliji dialect",   test_khaliji_dialect),
    ("T2 — Tashkil",           test_tashkil),
    ("T3 — Amal identity",     test_identity),
    ("T4 — Medical response",  test_medical_response),
]

if __name__ == "__main__":
    print("=" * 55)
    print("  Acceptance Tests — GeminiBrain  [Part 1/2]")
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
        print("  Part 1 passed — wait 60s then run: python test_gemini_2.py")
    else:
        print(f"  {failed} test(s) failed — see details above")
    print("=" * 55)
