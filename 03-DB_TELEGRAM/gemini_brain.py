"""
gemini_brain.py
───────────────
Gemini Flash LLM brain.
set_context() injects patient profile + protocol context before each turn.
voice_agent_llm.py is responsible for fetching and passing context (RAG).
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_BASE = (
    "أنتِ أَمَلْ، مساعدة ذكية في عيادة مدرسة بنات في قطر. "
    "تتحدثين باللهجة القطرية بلطف وحنان. "
    "تعملين تحت إشراف الممرضة نورة.\n\n"
    "قواعد:\n"
    "- ردودك قصيرة (2-3 جمل) باللهجة القطرية\n"
    "- لا تشخصين طبياً — أنتِ تساعدين وتوجهين فقط\n"
    "- عند الطوارئ: أبلغي الممرضة فوراً وقولي ذلك بوضوح\n"
    "- لا تذكري أرقام بروتوكولات للطالبة\n"
    "- تذكري الحساسيات والأمراض المزمنة عند الإجابة\n"
)


class GeminiBrain:

    def __init__(self):
        self.client  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.history = []
        self._system = _SYSTEM_BASE   # enriched by set_context()

    def set_context(self, patient_ctx: str = "", protocol_ctx: str = ""):
        """
        Inject patient profile + protocol context into system prompt.
        Called by voice_agent_llm.py before each think() call.
        """
        extra = ""
        if patient_ctx:
            extra += f"\n=== ملف الطالبة ===\n{patient_ctx}\n"
        if protocol_ctx:
            extra += f"\n=== البروتوكولات المناسبة ===\n{protocol_ctx}\n"
        self._system = _SYSTEM_BASE + extra

    def think(self, user_input: str) -> str:
        self.history.append({
            "role": "user",
            "parts": [{"text": user_input}]
        })
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=self._system,
                temperature=0.7,
                max_output_tokens=250,
            ),
        )
        reply = response.text.strip()
        self.history.append({
            "role": "model",
            "parts": [{"text": reply}]
        })
        return reply
