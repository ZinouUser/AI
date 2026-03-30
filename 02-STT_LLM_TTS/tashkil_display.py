# tashkil_display.py
import arabic_reshaper
from bidi.algorithm import get_display


class TashkilDisplay:
    """
    Affichage arabe en terminal (WSL/VSCode) et vérification du tashkil.
    
    - print_arabic()  : affiche avec bonnes formes de lettres (reshape+bidi)
    - show_tashkil()  : vérifie le ratio harakat/mot + écrit dans le log
    - Le log (tashkil_debug.log) contient le texte brut pour VS Code
    """

    LOG_FILE     = "tashkil_debug.log"
    _TASHKIL_SET = set("ًٌٍَُِّْٰٖٗ")

    # Seuil : au moins 2 harakat par mot en moyenne
    # (pleinement vowellisé ≈ 3–5 harakat/mot)
    _MIN_RATIO = 2.0

    # ── FLAG GLOBAL ─────────────────────────────────────────────────────
    enabled = True   # False => toutes les methodes deviennent no-op
    # ────────────────────────────────────────────────────────────────────

    def print_arabic(self, label: str, text: str) -> None:
        """Affiche le texte arabe avec bonnes formes de lettres."""
        reshaped  = arabic_reshaper.reshape(text)
        displayed = get_display(reshaped)
        print(f"{label}{displayed}")

    def count_harakat(self, text: str) -> int:
        """Compte les signes de tashkil dans le texte."""
        return sum(1 for c in text if c in self._TASHKIL_SET)

    def show_tashkil(self, label: str, text: str) -> None:
        """
        Vérifie le tashkil et affiche dans le terminal.
        
        Seuil : ratio harakat / nombre de mots >= 2.0
          < 2.0 → ✗  (tashkil incomplet ou absent)
          ≥ 2.0 → ✓  (tashkil suffisant pour TTS)
        
        Le texte brut (avec tashkil) est écrit dans tashkil_debug.log
        pour vérification dans VS Code.
        """
        n      = self.count_harakat(text)
        words  = [w for w in text.split() if w.strip("؟!،. ")]
        n_words = max(1, len(words))
        ratio  = n / n_words
        ok     = "✓" if ratio >= self._MIN_RATIO else "✗"

        # Terminal : compte + ratio (tashkil non visible en terminal WSL)
        displayed = get_display(text)   # bidi sans reshape → tashkil préservé
        print(f"  {label}[{n} harakat, {ratio:.1f}/mot {ok}]: {displayed}")

        # Log fichier : texte brut lisible dans VS Code
        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{label}[{n} h, {ratio:.1f}/mot {ok}] {text}\n")

    def init_log(self) -> None:
        """Efface le log au démarrage d'une session."""
        with open(self.LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== tashkil_debug.log ===\n")
