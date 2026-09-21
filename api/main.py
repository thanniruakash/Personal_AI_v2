import json
import os
import re
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loader import KnowledgeLoader
from core.search import KnowledgeSearch

try:
    from google import genai
except ImportError:
    genai = None

app = FastAPI(title="Personal AI v2 API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Knowledge + Gemini setup
# ------------------------------------------------------------
knowledge = KnowledgeLoader().load()
search_engine = KnowledgeSearch(knowledge)
LOCAL_THRESHOLD = 3000
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_KEY_NAMES = ["GEMINI_API_KEY", "KEY2", "KEY3", "KEY4"]


def build_gemini_clients():
    clients = []
    if genai is None:
        return clients

    for key_name in GEMINI_KEY_NAMES:
        key = os.environ.get(key_name)
        if not key:
            continue
        try:
            clients.append((key_name, genai.Client(api_key=key)))
        except Exception:
            pass
    return clients


gemini_clients = build_gemini_clients()


# ------------------------------------------------------------
# Request models
# ------------------------------------------------------------
class HistoryMessage(BaseModel):
    role: str
    text: str = ""


class ChatRequest(BaseModel):
    message: str
    mode: str = "auto"
    conversation_id: str | None = None
    history: list[HistoryMessage] = Field(default_factory=list)


# ------------------------------------------------------------
# Local knowledge
# ------------------------------------------------------------
def local_answer(message):
    results = search_engine.search(message)

    if not results:
        return None, 0, None, []

    best = results[0]
    answer = best.get("definition", "").strip()
    if not answer:
        answer = best.get("content", "").strip()

    # Keep the metadata small and useful for the frontend search/history UI.
    evidence = []
    for item in results[:5]:
        evidence.append({
            "topic": item.get("topic"),
            "subject": item.get("subject"),
            "score": int(item.get("_score", 0)),
        })

    return (
        answer,
        int(best.get("_score", 0)),
        best.get("topic") or None,
        evidence,
    )


# ------------------------------------------------------------
# Gemini prompt / context
# ------------------------------------------------------------
SYSTEM_INSTRUCTION = """You are Personal AI, a friendly student-focused assistant.

Answer the user's latest question directly and naturally.

Rules:
- Simple definition or normal question: answer briefly in 1-3 clear sentences.
- If the user asks to explain: give a clear moderate explanation.
- If the user asks for detailed/full information: give a detailed but focused answer.
- If the user asks for 2 marks: give a concise 2-mark answer.
- If the user asks for 5 marks: give a properly structured 5-mark answer.
- If the user asks for 10 marks: give a complete exam-style 10-mark answer with suitable headings, key points, examples or equations when relevant, and a short conclusion.
- Use simple language suitable for a college student.
- Continue naturally from the conversation when the latest question refers to something earlier.
- Do not claim that you searched the web unless web search was actually performed.
- Do not invent sources, facts, search results, or personal memories.
- Use Markdown when useful. For mathematics, use LaTeX such as $x^2$ or $$x^2+y^2$$.
- Never add unnecessary information.
"""


def build_gemini_prompt(message, history):
    recent = []
    for item in history[-12:]:
        role = "User" if item.role == "user" else "Personal AI"
        text = item.text.strip()
        if text:
            recent.append(f"{role}: {text}")

    context = "\n".join(recent)
    if context:
        return f"{SYSTEM_INSTRUCTION}\n\nConversation context:\n{context}\n\nUser's latest question:\n{message}"

    return f"{SYSTEM_INSTRUCTION}\n\nUser's latest question:\n{message}"


def gemini_text(message, history):
    prompt = build_gemini_prompt(message, history)

    for key_name, client in gemini_clients:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip(), key_name
        except Exception:
            continue

    return None, None


def gemini_stream(message, history):
    """Yield text chunks from Gemini using the official streaming API."""
    prompt = build_gemini_prompt(message, history)

    for key_name, client in gemini_clients:
        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            emitted = False
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    emitted = True
                    yield text, key_name

            if emitted:
                return
        except Exception:
            continue

    return


def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Personal AI v2",
        "version": "2.0.0",
        "knowledge_topics": len(search_engine.topic_index),
        "gemini_clients": len(gemini_clients),
        "gemini_model": GEMINI_MODEL,
        "streaming": bool(gemini_clients),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    mode = request.mode.lower().strip()

    if not message:
        return {
            "answer": "Please type a question.",
            "source": "local",
            "fallback": False,
        }

    history = request.history[-12:]

    # --------------------------------------------------------
    # Explicit Local mode
    # --------------------------------------------------------
    if mode == "local":
        answer, score, topic, evidence = local_answer(message)
        final_answer = answer or "I could not find a good answer in my local knowledge."
        return {
            "answer": final_answer,
            "source": "local",
            "fallback": False,
            "score": score,
            "topic": topic,
            "evidence": evidence,
            "conversation_id": request.conversation_id,
        }

    # --------------------------------------------------------
    # Explicit Gemini mode — real server-side streaming
    # --------------------------------------------------------
    if mode == "gemini":
        def stream_gemini_only():
            yield sse("meta", {
                "source": "gemini",
                "fallback": False,
                "conversation_id": request.conversation_id,
            })

            answer_parts = []
            used_key = None
            for chunk, key_name in gemini_stream(message, history):
                used_key = key_name
                answer_parts.append(chunk)
                yield sse("chunk", {"text": chunk})

            answer = "".join(answer_parts).strip()
            if not answer:
                yield sse("error", {
                    "message": "Gemini is currently unavailable.",
                    "source": "gemini",
                })
                return

            yield sse("done", {
                "answer": answer,
                "source": "gemini",
                "fallback": False,
                "gemini_key": used_key,
                "conversation_id": request.conversation_id,
            })

        return StreamingResponse(
            stream_gemini_only(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --------------------------------------------------------
    # Auto routing
    # --------------------------------------------------------
    lower_message = message.lower()

    marks_request = bool(re.search(
        r"\b(?:1|2|5|10)\s*marks?\b", lower_message
    ))

    changing_information = bool(re.search(
        r"\b(current|today|now|latest|recent|daily|monthly|"
        r"this week|this month|this year|current affairs|news|"
        r"price|prices|cost|rate|rates|exchange rate|currency|"
        r"gold|silver|stock|stocks|share price|petrol|diesel|"
        r"weather|temperature)\b", lower_message
    ))

    # Strong local match is used immediately for ordinary questions.
    answer, score, topic, evidence = local_answer(message)
    if not marks_request and not changing_information and answer and score >= LOCAL_THRESHOLD:
        return {
            "answer": answer,
            "source": "local",
            "fallback": False,
            "score": score,
            "topic": topic,
            "evidence": evidence,
            "conversation_id": request.conversation_id,
        }

    # Otherwise stream Gemini. This includes exam questions and changing information.
    def stream_auto():
        yield sse("meta", {
            "source": "gemini",
            "fallback": bool(answer),
            "local_score": score,
            "local_topic": topic,
            "conversation_id": request.conversation_id,
        })

        answer_parts = []
        used_key = None
        for chunk, key_name in gemini_stream(message, history):
            used_key = key_name
            answer_parts.append(chunk)
            yield sse("chunk", {"text": chunk})

        final_answer = "".join(answer_parts).strip()
        if final_answer:
            yield sse("done", {
                "answer": final_answer,
                "source": "gemini",
                "fallback": bool(answer),
                "local_score": score,
                "local_topic": topic,
                "evidence": evidence,
                "gemini_key": used_key,
                "conversation_id": request.conversation_id,
            })
            return

        # If Gemini is unavailable, return the local answer when we have one.
        if answer:
            yield sse("done", {
                "answer": answer,
                "source": "local",
                "fallback": True,
                "score": score,
                "topic": topic,
                "evidence": evidence,
                "conversation_id": request.conversation_id,
            })
            return

        yield sse("error", {
            "message": "Online information is currently unavailable.",
            "source": "gemini",
        })

    return StreamingResponse(
        stream_auto(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
