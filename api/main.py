import os
import json
import sys
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loader import KnowledgeLoader
from core.search import KnowledgeSearch

try:
    from google import genai
except ImportError:
    genai = None

app = FastAPI(title='Personal AI v2 API', version='1.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 💾 PERSISTENT CHAT MEMORY
# ============================================================

MEMORY_DIR = PROJECT_ROOT / "data"
MEMORY_FILE = MEMORY_DIR / "chat_memory.json"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def load_chat_memory():
    try:
        if not MEMORY_FILE.exists():
            return []

        data = json.loads(
            MEMORY_FILE.read_text(encoding="utf-8")
        )

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_chat_memory(history):
    MEMORY_FILE.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


chat_memory = load_chat_memory()

knowledge = KnowledgeLoader().load()
search_engine = KnowledgeSearch(knowledge)
LOCAL_THRESHOLD = 5
GEMINI_MODEL = 'gemini-3.6-flash'
GEMINI_KEY_NAMES = ['GEMINI_API_KEY', 'KEY2', 'KEY3', 'KEY4']

def build_gemini_clients():
    clients = []
    if genai is None:
        return clients
    for key_name in GEMINI_KEY_NAMES:
        key = os.environ.get(key_name)
        if key:
            try:
                clients.append((key_name, genai.Client(api_key=key)))
            except Exception:
                pass
    return clients

gemini_clients = build_gemini_clients()

class ChatRequest(BaseModel):
    message: str
    mode: str = 'auto'

def local_answer(message):
    results = search_engine.search(message)

    if not results:
        return None, 0

    best = results[0]
    answer = best.get('definition', '').strip()

    if not answer:
        answer = best.get('content', '').strip()

    return answer, int(best.get('_score', 0))

def gemini_answer(message):
    instruction = """You are Personal AI, a student-friendly assistant.

Decide the answer length from the user's request.

Rules:
- Simple definition or normal question: answer briefly in 1-3 clear sentences.
- If the user asks to explain: give a clear moderate explanation.
- If the user asks for detailed/full information: give a detailed but focused answer.
- If the user asks for 2 marks: give a concise 2-mark answer.
- If the user asks for 5 marks: give a properly structured 5-mark answer.
- If the user asks for 10 marks: give a complete exam-style 10-mark answer with suitable headings, key points, examples or equations when relevant, and a short conclusion.
- Never add unnecessary information.
- Use simple language suitable for a student.
- Answer the question directly.

User question: """ + message

    for key_name, client in gemini_clients:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=instruction
            )
            text = getattr(response, 'text', None)
            if text and text.strip():
                return text.strip(), key_name
        except Exception:
            continue

    return None, None

@app.get('/api/health')
def health():
    return {'status':'ok','service':'Personal AI v2','knowledge_topics':len(search_engine.topic_index),'gemini_clients':len(gemini_clients),'gemini_model':GEMINI_MODEL}

@app.post('/api/chat')
def chat(request: ChatRequest):
    message = request.message.strip()
    mode = request.mode.lower().strip()

    if not message:
        return {
            'answer': 'Please type a question.',
            'source': 'local',
            'fallback': False
        }

    def remember_chat(answer, source, fallback=False, score=None):
        entry = {
            'question': message,
            'answer': answer,
            'source': source,
            'fallback': fallback
        }

        if score is not None:
            entry['score'] = score

        chat_memory.append(entry)

        # Keep only the latest 1000 exchanges.
        if len(chat_memory) > 1000:
            del chat_memory[:-1000]

        save_chat_memory(chat_memory)

    if mode == 'gemini':
        answer, _ = gemini_answer(message)

        if answer:
            remember_chat(answer, 'gemini', False)
            return {
                'answer': answer,
                'source': 'gemini',
                'fallback': False
            }

        answer = 'Gemini is currently unavailable.'
        remember_chat(answer, 'gemini', False)

        return {
            'answer': answer,
            'source': 'gemini',
            'fallback': False,
            'error': True
        }

    # ========================================================
    # SMART ROUTING
    # ========================================================

    lower_message = message.lower()

    # Exam questions and changing information always use Gemini.
    marks_request = bool(re.search(
        r'\b(?:1|2|5|10)\s*marks?\b',
        lower_message
    ))

    changing_information = bool(re.search(
        r'\b(current|today|now|latest|recent|daily|monthly|'
        r'this week|this month|this year|current affairs|news|'
        r'price|prices|cost|rate|rates|exchange rate|currency|'
        r'gold|silver|stock|stocks|share price|petrol|diesel|'
        r'weather|temperature)\b',
        lower_message
    ))

    if mode == 'auto' and (marks_request or changing_information):

        gemini_text, _ = gemini_answer(message)

        if gemini_text:
            remember_chat(gemini_text, 'gemini', False)

            return {
                'answer': gemini_text,
                'source': 'gemini',
                'fallback': False
            }

        return {
            'answer': 'Online information is currently unavailable.',
            'source': 'gemini',
            'fallback': False,
            'error': True
        }

    # Normal questions: use local knowledge only when the
    # search match is strong enough.
    answer, score = local_answer(message)

    if mode == 'local':
        final_answer = (
            answer or
            'I could not find a good answer in my local knowledge.'
        )

        remember_chat(final_answer, 'local', False, score)

        return {
            'answer': final_answer,
            'source': 'local',
            'fallback': False,
            'score': score
        }

    if answer and score >= 3000:
        remember_chat(answer, 'local', False, score)

        return {
            'answer': answer,
            'source': 'local',
            'fallback': False,
            'score': score
        }

    gemini_text, _ = gemini_answer(message)

    if gemini_text:
        remember_chat(gemini_text, 'gemini', True, score)

        return {
            'answer': gemini_text,
            'source': 'gemini',
            'fallback': True,
            'local_score': score
        }

    return {
        'answer': 'I could not find a reliable answer for that yet.',
        'source': 'gemini',
        'fallback': False,
        'error': True
    }