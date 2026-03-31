"""
QatariDialect — Deterministic post-processing pipeline for Qatari Gulf Arabic TTS.
Applied to Gemini output before sending to ElevenLabs.
"""

# qatari_dialect.py
import re


class QatariDialect:
    """
    Post-processing deterministe : Gemini (arabe standard)
    => phonetique qatarie / khaliji.

    Pipeline :
      1. fix_initial_sukun   - pas de sukun initial
      2. tanwin_to_sukun     - tanwin -> sukun
      3. ta_marbuta_sukun    - ta marbuta toujours muette  [NOUVEAU]
      4. hamza_to_ya         - hamza apres kasra -> ya     [NOUVEAU]
      5. diphthong_ay        - ay -> ee  (bayt -> beet)
      6. diphthong_aw        - aw -> oo  (yawm -> yoom)   [NOUVEAU]
      7. remove_final_hamza  - hamza finale elidee
      8. hamza_wasl          - al- en liaison
      9. khaliji_phonetics   - qaf -> gaf
    """

    _TASHKIL         = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0656\u0657"
    _Q_TO_G          = str.maketrans("\u0642", "\u06AF")           # ق → گ
    _KEEP_Q          = frozenset({"\u0627\u0644\u0642\u0631\u0622\u0646",   # القرآن
                                   "\u0642\u0631\u0622\u0646",              # قرآن
                                   "\u0627\u0644\u0641\u0631\u0642\u0627\u0646",  # الفرقان
                                   "\u0642\u0644",                          # قل
                                   "\u0642\u0631\u0623"})                   # قرأ
    _TANWIN_TO_SUKUN = str.maketrans({"\u064B": "\u0652",          # ً → ْ
                                       "\u064C": "\u0652",          # ٌ → ْ
                                       "\u064D": "\u0652"})         # ٍ → ْ
    _HAMZA_CHARS     = frozenset({"\u0621", "\u0623", "\u0625",    # ء أ إ
                                   "\u0624", "\u0626"})             # ؤ ئ

    # Regex — sans r'' pour que \u soit traite comme Unicode par Python
    _RE_DIPH_AY = re.compile("\u064E\u064A\u0652(?=[\u0600-\u06FF\u0671])")
    _RE_DIPH_AW = re.compile("\u064E\u0648\u0652(?=[\u0600-\u06FF\u0671])")

    # ------------------------------------------------------------------ #
    #  Pipeline                                                            #
    # ------------------------------------------------------------------ #

    def process(self, text: str) -> str:
        text = self.fix_initial_sukun(text)
        text = self.tanwin_to_sukun(text)
        text = self.ta_marbuta_sukun(text)
        text = self.hamza_to_ya(text)
        text = self.diphthong_ay(text)
        text = self.diphthong_aw(text)
        text = self.remove_final_hamza(text)
        text = self.hamza_wasl(text)
        text = self.khaliji_phonetics(text)
        return text

    # ------------------------------------------------------------------ #

    def fix_initial_sukun(self, text: str) -> str:
        """Mot commencant par sukun -> fatha prosthétique."""
        words = text.split()
        result = []
        for word in words:
            if len(word) >= 2 and word[1] == "\u0652":
                word = word[0] + "\u064E" + word[2:]
            result.append(word)
        return " ".join(result)

    def tanwin_to_sukun(self, text: str) -> str:
        """Tanwin banni en qatari -> sukun.
        Cas special : alef-siege + tanwin-fath -> sukun seul.
        """
        text = text.replace("\u0627\u064B", "\u0652")   # اً → ْ
        return text.translate(self._TANWIN_TO_SUKUN)

    def ta_marbuta_sukun(self, text: str) -> str:
        """Ta marbuta toujours muette en qatari -> sukun obligatoire.
        Remplace fatha/damma/kasra sur ta marbuta par sukun.
        """
        return re.sub("[\u0629][\u064E\u064F\u0650]", "\u0629\u0652", text)

    def hamza_to_ya(self, text: str) -> str:
        """Hamza apres kasra -> ya (tashil).
        Kasra + hamza-sur-ya -> kasra + ya.
        """
        return text.replace("\u0650\u0626", "\u0650\u064A")

    def diphthong_ay(self, text: str) -> str:
        """Diphtongue /ay/ -> son long /ii/ en qatari.
        Fatha + ya + sukun devant consonne -> kasra + ya (sans sukun).
        Ex : bayt -> beet, khayr -> kheer.
        """
        return self._RE_DIPH_AY.sub("\u0650\u064A", text)

    def diphthong_aw(self, text: str) -> str:
        """Diphtongue /aw/ -> son long /uu/ en qatari.
        Fatha + waw + sukun devant consonne -> damma + waw (sans sukun).
        Ex : yawm -> yoom.
        """
        return self._RE_DIPH_AW.sub("\u064F\u0648", text)

    def remove_final_hamza(self, text: str) -> str:
        """Hamza finale elidee en qatari (toutes formes).
        Hamza + sukun en fin de mot -> supprime les deux.
        """
        words = text.split()
        result = []
        for word in words:
            if len(word) >= 2 and word[-1] == "\u0652" and word[-2] in self._HAMZA_CHARS:
                word = word[:-2]
            elif word and word[-1] in self._HAMZA_CHARS:
                word = word[:-1]
            result.append(word)
        return " ".join(result)

    def hamza_wasl(self, text: str) -> str:
        """Alef de l'article silent en liaison apres haraka."""
        return re.sub(
            "([\u064E\u064F\u0650\u0652])\\s+\u0627\u0644",
            "\\1 \u0671\u0644",
            text
        )

    def khaliji_phonetics(self, text: str) -> str:
        """Qaf -> Gaf (sauf termes religieux)."""
        words = text.split()
        result = []
        for word in words:
            bare = word.strip(self._TASHKIL + "?!.,\u060C\u061F ")
            if bare in self._KEEP_Q:
                result.append(word)
            else:
                result.append(word.translate(self._Q_TO_G))
        return " ".join(result)
