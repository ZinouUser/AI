"""
db_protocols.py
───────────────
ChromaDB vector store — medical protocols for school nurse AI.
Semantic search via Google gemini-embedding-001.

8 protocols covering common school nurse scenarios:
  P01 — رشح خفيف       / Mild cold
  P02 — حساسية اللاكتوز / Lactose intolerance reaction
  P03 — نوبة ربو        / Asthma attack
  P04 — نقص سكر         / Hypoglycemia (type 1 diabetes)
  P05 — إغماء           / Syncope
  P06 — ألم في البطن    / Abdominal pain
  P07 — جرح بسيط        / Minor wound / bleeding
  P08 — حمى             / Fever

Schema convention (language-agnostic):
  metadata keys without _en → Arabic (school language)
  metadata keys with    _en → English
"""

import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY   = os.getenv("GEMINI_API_KEY")
CHROMA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "protocols_chroma")
COLLECTION   = "medical_protocols"
EMBED_MODEL  = "gemini-embedding-001"


# ── Protocol definitions ───────────────────────────────────────────────────────

_PROTOCOLS = [
    {
        "id":         "P01",
        "title":      "رشح خفيف",
        "title_en":   "Mild Cold",
        "urgency":    "routine",
        "tags":       "رشح,سيلان الأنف,عطاس,ازدحام أنفي,سعال خفيف",
        "tags_en":    "cold,runny nose,sneezing,nasal congestion,mild cough",
        "content":    """
بروتوكول P01 — رشح خفيف

الأعراض الشائعة:
- سيلان الأنف أو احتقان
- عطاس متكرر
- سعال خفيف
- حرارة خفيفة (أقل من 38°C)
- تعب عام خفيف

تقييم الممرضة:
1. قيسي الحرارة — إذا أكثر من 38°C طبّقي بروتوكول P08
2. تأكدي من عدم وجود ضيق تنفس (إذا وجد → P03)
3. اسألي عن مدة الأعراض

الإجراءات:
- أريحي الطالبة في غرفة العيادة
- أعطيها مناديل وعلّميها غسل اليدين
- وفّري ماء دافئ أو شاي بالليمون
- إذا طلبت مسكن → تواصلي مع ولي الأمر أولاً

التواصل:
- أبلغي ولي الأمر إذا استمرت الأعراض أكثر من يومين
- لا حاجة لإخراج الطالبة من الفصل إلا إذا كانت متعبة جداً

مستوى الطوارئ: روتين
""",
    },
    {
        "id":         "P02",
        "title":      "تفاعل حساسية اللاكتوز",
        "title_en":   "Lactose Intolerance Reaction",
        "urgency":    "monitor",
        "tags":       "حساسية,لاكتوز,ألبان,ألم بطن,انتفاخ,غثيان,إسهال,حساسية غذائية",
        "tags_en":    "allergy,lactose,dairy,stomach pain,bloating,nausea,diarrhea,food intolerance",
        "content":    """
بروتوكول P02 — تفاعل حساسية اللاكتوز

ملاحظة مهمة: هذا البروتوكول مخصص لحساسية اللاكتوز (عدم تحمّل)
وليس لحساسية الحليب المناعية (IgE) التي تستدعي بروتوكول طوارئ مختلف.

الأعراض الشائعة (تظهر خلال 30 دقيقة – ساعتين من تناول منتجات الألبان):
- ألم أو تشنج في البطن
- انتفاخ وغازات
- غثيان
- إسهال
- قرقرة في البطن

تقييم الممرضة:
1. اسألي: هل تناولت منتجات ألبان اليوم؟
2. تحققي من عدم وجود طفح جلدي أو ضيق تنفس (→ طوارئ فورية إذا وجد)
3. سجّلي الوقت وما أكلته

الإجراءات:
- أريحي الطالبة في وضع مريح
- لا تعطيها أي منتج ألبان
- الماء الدافئ يساعد على تخفيف التشنج
- معظم الأعراض تزول خلال 1-2 ساعة

التواصل:
- أبلغي ولي الأمر
- ذكّري الطالبة بتجنب منتجات الألبان في المقصف

مستوى الطوارئ: مراقبة
""",
    },
    {
        "id":         "P03",
        "title":      "نوبة ربو",
        "title_en":   "Asthma Attack",
        "urgency":    "urgent",
        "tags":       "ربو,ضيق تنفس,صفير,بخاخ,أزمة,تنفس صعب,لهاث",
        "tags_en":    "asthma,breathing difficulty,wheezing,inhaler,attack,shortness of breath",
        "content":    """
بروتوكول P03 — نوبة ربو

⚠️ بروتوكول طوارئ — تصرفي فوراً

الأعراض:
- صعوبة في التنفس أو ضيق
- صفير عند الزفير
- سعال مستمر
- شعور بضيق في الصدر
- في الحالات الشديدة: زرقة حول الشفاه

الإجراءات الفورية:
1. أبقي الطالبة جالسة منتصبة — لا تضعيها مستلقية
2. ابحثي عن بخاخ الربو في حقيبتها فوراً
3. ساعديها على استخدام البخاخ (2 نفخة salbutamol)
4. افتحي النوافذ لتهوية الغرفة
5. اتصلي بالإسعاف إذا لم يتحسن خلال 5 دقائق

بعد استخدام البخاخ:
- راقبي التنفس لمدة 15 دقيقة
- إذا تحسّن: أبلغي ولي الأمر وسجّلي الحادثة
- إذا لم يتحسن أو ساء: اتصلي بالإسعاف 999

التواصل:
- أبلغي إدارة المدرسة فوراً
- اتصلي بولي الأمر فوراً
- لا تتركي الطالبة وحدها

مستوى الطوارئ: طارئ — تصرف فوري
""",
    },
    {
        "id":         "P04",
        "title":      "نقص سكر الدم — سكري نوع 1",
        "title_en":   "Hypoglycemia — Type 1 Diabetes",
        "urgency":    "urgent",
        "tags":       "سكري,نقص سكر,دوخة,تعرق,ارتجاف,جوع مفاجئ,أنسولين,هبوط سكر",
        "tags_en":    "diabetes,hypoglycemia,dizziness,sweating,trembling,sudden hunger,insulin,low blood sugar",
        "content":    """
بروتوكول P04 — نقص سكر الدم (سكري نوع 1)

⚠️ بروتوكول طوارئ — تصرفي فوراً

الأعراض التحذيرية:
- دوخة أو إغماء خفيف
- تعرق مفاجئ
- ارتجاف أو رعشة في اليدين
- جوع شديد ومفاجئ
- شحوب الوجه
- تشوش أو صعوبة في التركيز
- في الحالات الشديدة: فقدان الوعي

الإجراءات الفورية (إذا كانت واعية):
1. أعطيها 15-20 غرام سكر سريع الامتصاص:
   - نصف كوب عصير برتقال أو
   - 3-4 حبات حلوى أو
   - كوب ماء مع ملعقة سكر
2. أجلسيها وانتظري 15 دقيقة
3. راقبي تحسن الأعراض
4. بعد التحسن: أعطيها وجبة خفيفة (بسكويت + جبن)

إذا فقدت الوعي:
- لا تعطيها أي شيء عن طريق الفم
- اتصلي بالإسعاف 999 فوراً
- ضعيها في وضع الإفاقة (على الجانب)

التواصل:
- أبلغي ولي الأمر فوراً
- ابحثي عن قلم الأنسولين في حقيبتها
- سجّلي الوقت ومستوى السكر إذا كان عندها جهاز قياس

مستوى الطوارئ: طارئ — تصرف فوري
""",
    },
    {
        "id":         "P05",
        "title":      "إغماء أو فقدان وعي",
        "title_en":   "Syncope / Loss of Consciousness",
        "urgency":    "urgent",
        "tags":       "إغماء,فقدان وعي,سقوط,دوخة,شحوب,وقوف مفاجئ,تعب,ضعف",
        "tags_en":    "syncope,fainting,loss of consciousness,dizziness,pallor,sudden standing,weakness",
        "content":    """
بروتوكول P05 — إغماء أو فقدان الوعي

⚠️ بروتوكول طوارئ

أعراض تحذيرية قبل الإغماء:
- دوخة مفاجئة
- شحوب الوجه
- تعرق بارد
- ضبابية في الرؤية
- طنين في الأذنين
- ضعف في الساقين

الإجراءات الفورية:
1. إذا شعرت بالإغماء قبل السقوط:
   - أجلسيها فوراً أو ضعيها على الأرض بأمان
   - ارفعي قدميها فوق مستوى الرأس (20-30 سم)

2. إذا فقدت الوعي:
   - تأكدي من سلامة مجرى التنفس
   - لا تتركيها وحدها أبداً
   - اتصلي بالإسعاف 999 إذا لم تستعد وعيها خلال دقيقة

بعد الاستعادة:
- أبقيها مستلقية لمدة 10 دقائق على الأقل
- لا تسمحي لها بالوقوف فجأة
- أعطيها ماء بارداً
- افحصي إذا أصيبت بجرح عند السقوط

ملاحظة للطالبات ذوات التاريخ المرضي:
- الإغماء المتكرر يستدعي إشعار الأهل فوراً
- سجّلي كل حادثة مع التوقيت والظروف

التواصل:
- أبلغي ولي الأمر فوراً
- أبلغي الإدارة
- سجّلي الحادثة في ملف الطالبة

مستوى الطوارئ: طارئ
""",
    },
    {
        "id":         "P06",
        "title":      "ألم في البطن",
        "title_en":   "Abdominal Pain",
        "urgency":    "monitor",
        "tags":       "ألم بطن,تشنج,مغص,غثيان,قيء,معدة,هضم,دورة شهرية,إمساك",
        "tags_en":    "abdominal pain,cramps,nausea,vomiting,stomach,digestion,menstrual,constipation",
        "content":    """
بروتوكول P06 — ألم في البطن

الأعراض:
- ألم أو تشنج في منطقة البطن
- غثيان مع أو بدون قيء
- انتفاخ
- قد يكون مصحوباً بدورة شهرية مؤلمة

تقييم الممرضة:
1. حددي موقع الألم (أعلى / أسفل / يمين / يسار / منتشر)
2. اسألي: هل الألم مستمر أم على شكل تشنجات؟
3. اسألي: هل أكلت اليوم؟ وماذا أكلت؟
4. اسألي: هل لديها دورة شهرية؟
5. تحققي من الحرارة

⚠️ علامات خطر — اتصلي بالإسعاف فوراً إذا:
- الألم شديد جداً ولا يحتمل
- الألم في الجانب الأيمن السفلي (احتمال التهاب الزائدة)
- مصحوب بحمى فوق 39°C
- قيء متكرر لا يتوقف

الإجراءات الروتينية:
- أريحيها في وضع مريح (مستلقية على جانبها)
- الكمادة الدافئة على البطن تساعد
- لا تعطيها مسكنات بدون إذن ولي الأمر

التواصل:
- أبلغي ولي الأمر إذا استمر الألم أكثر من 30 دقيقة

مستوى الطوارئ: مراقبة
""",
    },
    {
        "id":         "P07",
        "title":      "جرح بسيط أو نزيف طفيف",
        "title_en":   "Minor Wound / Superficial Bleeding",
        "urgency":    "routine",
        "tags":       "جرح,نزيف,خدش,قطع,جلد,ضمادة,تطهير,إسعافات أولية",
        "tags_en":    "wound,bleeding,scratch,cut,skin,bandage,disinfect,first aid",
        "content":    """
بروتوكول P07 — جرح بسيط أو نزيف طفيف

الأعراض:
- جرح سطحي أو خدش
- نزيف خفيف
- كدمة

تقييم الممرضة:
1. تقييم عمق الجرح — إذا كان عميقاً أو لا يتوقف النزيف → أبلغي الإدارة
2. تحققي من نظافة الجرح (تراب / زجاج)
3. اسألي عن آخر تطعيم تيتانوس

⚠️ حالات تحتاج إسعاف:
- نزيف لا يتوقف بعد 10 دقائق ضغط مباشر
- جرح عميق يحتاج خياطة
- إصابة في الرأس أو العين

الإجراءات:
1. اغسلي يديك قبل التعامل مع الجرح
2. اضغطي على الجرح بقطعة شاش نظيفة لـ 5-10 دقائق
3. نظّفي بمحلول مطهر (بيتادين أو كلورهيكسيدين)
4. ضعي ضمادة مناسبة
5. إذا كانت الطالبة تعاني من قلق → طمئنيها أولاً

التواصل:
- أبلغي ولي الأمر بحادثة الجرح
- سجّلي الحادثة

مستوى الطوارئ: روتين
""",
    },
    {
        "id":         "P08",
        "title":      "حمى",
        "title_en":   "Fever",
        "urgency":    "monitor",
        "tags":       "حمى,حرارة,ارتفاع حرارة,قشعريرة,صداع,ترمومتر,درجة حرارة",
        "tags_en":    "fever,temperature,high temperature,chills,headache,thermometer",
        "content":    """
بروتوكول P08 — حمى

الأعراض:
- حرارة فوق 37.5°C
- قشعريرة
- صداع
- تعب عام
- احمرار الوجه

تصنيف الحمى:
  37.5 – 38.0°C  : حمى خفيفة — مراقبة
  38.0 – 39.0°C  : حمى متوسطة — أبلغي ولي الأمر
  فوق 39.0°C     : حمى شديدة — إخراج من المدرسة فوراً

الإجراءات:
1. قيسي الحرارة بالترمومتر (تحت الإبط أو الجبهة)
2. أريحيها في مكان هادئ
3. شجّعيها على شرب الماء
4. كمادة باردة على الجبهة لتخفيف الانزعاج

⚠️ اتصلي بالإسعاف فوراً إذا:
- الحرارة فوق 40°C
- مصحوبة بتشنجات
- مصحوبة بطفح جلدي
- الطالبة فاقدة الوعي أو غير مستجيبة

التواصل:
- أبلغي ولي الأمر فوراً عند 38°C وما فوق
- لا تعطِ مخفض حرارة بدون إذن ولي الأمر
- الطالبة يجب أن تغادر المدرسة إذا كانت الحرارة فوق 38.5°C

مستوى الطوارئ: مراقبة
""",
    },
]


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Google gemini-embedding-001."""
    client = genai.Client(api_key=GEMINI_KEY)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]


def _embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    client = genai.Client(api_key=GEMINI_KEY)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


# ── Init ───────────────────────────────────────────────────────────────────────

def init_protocols(force: bool = False):
    """
    Create ChromaDB collection and embed all 8 protocols.

    force=True  → drop and recreate (useful after protocol edits)
    force=False → skip if collection already has documents
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if force:
        try:
            client.delete_collection(COLLECTION)
            print(f"  [Chroma] Collection '{COLLECTION}' dropped (force=True)")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if col.count() > 0 and not force:
        print(f"  [Chroma] '{COLLECTION}' already loaded ({col.count()} protocols)")
        return col

    print(f"  [Chroma] Embedding {len(_PROTOCOLS)} protocols "
          f"via {EMBED_MODEL}...")

    ids        = [p["id"]      for p in _PROTOCOLS]
    documents  = [p["content"] for p in _PROTOCOLS]
    metadatas  = [
        {
            "title":    p["title"],
            "title_en": p["title_en"],
            "urgency":  p["urgency"],
            "tags":     p["tags"],
            "tags_en":  p["tags_en"],
        }
        for p in _PROTOCOLS
    ]

    embeddings = _embed(documents)

    col.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"  [Chroma] ✓ {col.count()} protocols stored → {CHROMA_PATH}")
    return col


# ── Search ─────────────────────────────────────────────────────────────────────

def search_protocols(query: str, n_results: int = 2) -> list[dict]:
    """
    Semantic search: return top-n protocols matching the query.

    Returns list of dicts:
      { id, title, title_en, urgency, tags, tags_en, content, distance }
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col    = client.get_collection(COLLECTION)

    query_embedding = _embed_query(query)

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "id":       results["ids"][0][i],
            "title":    meta["title"],
            "title_en": meta["title_en"],
            "urgency":  meta["urgency"],
            "tags":     meta["tags"],
            "tags_en":  meta["tags_en"],
            "content":  results["documents"][0][i],
            "distance": round(results["distances"][0][i], 4),
        })

    return hits


def protocol_context(query: str, n_results: int = 2) -> str:
    """
    Return top-n protocol contents as a single string for LLM injection.
    """
    hits = search_protocols(query, n_results=n_results)
    if not hits:
        return ""
    sections = []
    for h in hits:
        sections.append(
            f"=== {h['id']} — {h['title']} (urgency: {h['urgency']}) ===\n"
            f"{h['content'].strip()}"
        )
    return "\n\n".join(sections)
