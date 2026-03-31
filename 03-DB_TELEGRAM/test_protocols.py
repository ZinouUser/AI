"""
test_protocols.py
─────────────────
Interactive semantic search test for db_protocols.py
  1. Prints all loaded protocols
  2. Shows a numbered symptom menu
  3. User picks a number → semantic search → top-2 results
"""

import sys
import os
import arabic_reshaper
from bidi.algorithm import get_display

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_protocols import init_protocols, search_protocols, _PROTOCOLS


def ar(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


# ── Symptom menu ───────────────────────────────────────────────────────────────

_SYMPTOMS = [
    ("1", "ضيق في التنفس أو صفير",          "عندي ضيق في التنفس وصفير"),
    ("2", "ألم في البطن أو غثيان",           "بطني يوجعني وأكلت جبن"),
    ("3", "دوخة وتعرق مفاجئ",               "أنا دايخة وتعرقت فجأة"),
    ("4", "جرح أو نزيف",                    "سقطت وعندي جرح في يدي"),
    ("5", "حرارة وصداع",                    "عندي حرارة وصداع"),
    ("6", "إغماء أو فقدان وعي",             "أغمي علي وصحيت"),
    ("7", "جوع شديد مفاجئ ورعشة في اليدين", "أحس بجوع شديد وأيدي ترتجف"),
]


def print_protocols():
    print("\n" + "═" * 55)
    # Arabic-only line → ar() on the whole string
    print("  " + ar("البروتوكولات الطبية المتاحة"))
    print("═" * 55)
    for p in _PROTOCOLS:
        urgency_label = {
            "routine": ar("روتين"),
            "monitor": ar("مراقبة"),
            "urgent":  ar("طارئ"),
        }.get(p["urgency"], p["urgency"])

        title = ar(p['title'])
        # Print ID left, title center, urgency right — avoid :<N on bidi strings
        print(f"  [{p['id']}]  {title}  —  {urgency_label}")

    print("═" * 55)


def print_menu():
    print("\n" + "─" * 55)
    # Mixed line: ar() only on the Arabic part
    print("  " + ar("اختاري الأعراض") + "  —  choose symptom:")
    print("─" * 55)
    for num, label, _ in _SYMPTOMS:
        print(f"  {num}.  {ar(label)}")
    print("─" * 55)


def print_results(hits: list[dict]):
    print()
    for h in hits:
        title = ar(h['title'])
        print(f"  [{h['id']}] {title}  "
              f"(urgency={h['urgency']}, dist={h['distance']})")
    print()


def main():
    print("\n=== test_protocols.py — semantic search ===")
    init_protocols()
    print_protocols()

    while True:
        print_menu()
        print("  (0 to exit)")
        choice = input("  → ").strip()

        if choice == "0":
            print("  Bye.\n")
            break

        match = next((s for s in _SYMPTOMS if s[0] == choice), None)
        if not match:
            print(f"  ✗ Invalid choice: '{choice}'")
            continue

        num, label, query = match
        print(f"\n  Query : {ar(query)}")
        print("  " + "─" * 50)

        hits = search_protocols(query, n_results=2)
        print_results(hits)


if __name__ == "__main__":
    main()
