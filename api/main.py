import json
import os
import re
import sys
import html
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.loader import KnowledgeLoader
from core.search import KnowledgeSearch
from calculator import Calculator

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
calculator = Calculator()
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


class LiveReportRequest(BaseModel):
    message: str
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
    answer_field = best.get("_answer_field")
    if answer_field:
        answer = str(best.get(answer_field, "")).strip()
    else:
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


def build_gemini_prompt(message, history, local_context=None):
    recent = []
    for item in history[-12:]:
        role = "User" if item.role == "user" else "Personal AI"
        text = item.text.strip()
        if text:
            recent.append(f"{role}: {text}")

    context = "\n".join(recent)
    parts = [SYSTEM_INSTRUCTION]

    if local_context:
        parts.append(
            "Relevant local knowledge found for this question. "
            "Use it when it is relevant to the user's question; never force "
            "it into the answer if it does not match the question:\n"
            + local_context
        )

    if context:
        parts.append(f"Conversation context:\n{context}")

    parts.append(f"User's latest question:\n{message}")
    return "\n\n".join(parts)


def gemini_text(message, history, local_context=None):
    prompt = build_gemini_prompt(message, history, local_context)

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


def gemini_stream(message, history, local_context=None):
    """Yield text chunks from Gemini using the official streaming API."""
    prompt = build_gemini_prompt(message, history, local_context)

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
# Deterministic math path
# ------------------------------------------------------------
def calculation_answer(message):
    """
    Return a calculator result when the message is clearly arithmetic.

    This intentionally runs before Local/Gemini routing so an LLM or fuzzy
    knowledge match can never be asked to perform basic arithmetic.
    """
    result = calculator.ask(message)
    if result is None:
        return None
    return result


# ------------------------------------------------------------
# Live query detection + real web retrieval (no invented sources)
# ------------------------------------------------------------
LIVE_PATTERNS = [
    re.compile(r"\b(latest|recent|current|currently|present|today|tonight|now|live|breaking|this week|this month|upcoming)\b", re.I),
    re.compile(r"\b(news|weather|forecast|showtimes?|cinema|movies?|theatres?|theaters?|f1|formula\s*1|grand prix|release date|release news|updates?)\b", re.I),
    re.compile(r"\bgta\s*(?:vi|6)\b", re.I),
]

def is_live_query(message: str) -> bool:
    text = (message or "").strip()
    return bool(text) and any(pattern.search(text) for pattern in LIVE_PATTERNS)

class _DDGParser(HTMLParser):
    """Extract DuckDuckGo HTML result links/snippets without extra dependencies."""
    def __init__(self):
        super().__init__()
        self.results=[]
        self._active=None
        self._capture=None
        self._text=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        classes=set((attrs.get('class') or '').split())
        if tag == 'a' and 'result__a' in classes:
            self._active={'title':'','url':attrs.get('href',''),'snippet':''}
            self._capture='title'; self._text=[]
        elif self._active and ('result__snippet' in classes):
            self._capture='snippet'; self._text=[]
    def handle_data(self, data):
        if self._active and self._capture:
            self._text.append(data)
    def handle_endtag(self, tag):
        if not self._active or not self._capture: return
        if self._capture=='title' and tag=='a':
            self._active['title']=' '.join(''.join(self._text).split())
            self._capture=None; self._text=[]
            # title link may be followed by a snippet; keep active until snippet or next result
        elif self._capture=='snippet' and tag in ('a','div','td'):
            self._active['snippet']=' '.join(''.join(self._text).split())
            self._capture=None; self._text=[]
            self._finish()
    def _finish(self):
        if self._active and self._active.get('title') and self._active.get('url'):
            self.results.append(self._active)
        self._active=None; self._capture=None; self._text=[]

def search_web(query: str, limit: int = 6):
    """Perform actual web search and return only links found in the search response."""
    url='https://html.duckduckgo.com/html/?' + urllib.parse.urlencode({'q': query})
    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 (compatible; PersonalAI/2.0; +https://example.com)'})
    with urllib.request.urlopen(req, timeout=8) as response:
        page=response.read(1_500_000).decode('utf-8','replace')
    parser=_DDGParser(); parser.feed(page)
    out=[]; seen=set()
    # HTML parser class structure can vary, so extract result anchors as a robust fallback.
    for item in parser.results:
        target=item.get('url','')
        if 'uddg=' in target:
            try: target=urllib.parse.parse_qs(urllib.parse.urlparse(target).query).get('uddg',[target])[0]
            except Exception: pass
        target=html.unescape(target)
        parsed=urllib.parse.urlparse(target)
        if parsed.scheme not in ('http','https') or not item.get('title') or target in seen: continue
        seen.add(target)
        out.append({'title':item['title'][:240], 'url':target, 'snippet':item.get('snippet','')[:700]})
        if len(out)>=limit: break
    if not out:
        # Regex fallback handles DuckDuckGo's common anchor markup if parser nesting differs.
        for match in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.I|re.S):
            target=html.unescape(match.group(1)); title=re.sub(r'<[^>]+>',' ',match.group(2)); title=' '.join(html.unescape(title).split())
            if target.startswith('//'): target='https:'+target
            if target.startswith('http') and title and target not in seen:
                seen.add(target); out.append({'title':title[:240],'url':target,'snippet':''})
                if len(out)>=limit: break
    return out

def live_report(message: str, history=None):
    sources=search_web(message, limit=6)
    if not sources:
        return {'answer':'I could not retrieve verifiable web results for this query right now. Please try again shortly.', 'source':'live', 'topic':message, 'sources':[], 'evidence':[], 'error':True}
    evidence='\n'.join(f"[{i+1}] {x['title']}\nURL: {x['url']}\nSnippet: {x['snippet']}" for i,x in enumerate(sources))
    history = history or []
    prompt=("You are Personal AI. Write a concise, useful Live Report based ONLY on the web search evidence below. "
            "State uncertainty when evidence is incomplete. Do not invent facts or claim more than sources support. "
            "Start with a one-paragraph summary, then key updates in bullets, then a short 'What to know' section. "
            "Mention dates when the sources provide them. Do not fabricate citations; source links will be shown separately.\n\n"
            f"User query: {message}\n\nSearch evidence:\n{evidence}")
    answer, used_key = gemini_text(prompt, history, None)
    if not answer:
        # Even when Gemini is unavailable, show the retrieved evidence instead of a blank answer.
        answer='Live Report\n\nI retrieved these web results, but the AI summary service is currently unavailable:\n\n' + '\n'.join(f"- **{x['title']}** — {x['snippet'] or 'Open the source for details.'}" for x in sources)
    return {'answer':answer.strip() or 'The live report could not be generated. Please try again.', 'source':'live', 'topic':message,
            'sources':[{'title':x['title'],'url':x['url']} for x in sources],
            'evidence':[{'title':x['title'],'url':x['url'],'snippet':x['snippet']} for x in sources],
            'gemini_key':used_key, 'generated_at':datetime.now(timezone.utc).isoformat(), 'fallback':not bool(used_key)}

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok" if gemini_clients else "degraded",
        "service": "Personal AI v2",
        "version": "2.0.0",
        "knowledge_topics": len(search_engine.topic_index),
        "gemini_clients": len(gemini_clients),
        "gemini_model": GEMINI_MODEL,
        "streaming": bool(gemini_clients),
        "checks": {
            "knowledge_loaded": bool(search_engine.topic_index),
            "gemini_configured": bool(gemini_clients),
            "gemini_key_count": len(gemini_clients),
        },
    }


@app.post("/api/live-report")
def live_report_route(request: LiveReportRequest):
    message = request.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"answer":"Please type a question for the Live Report.","source":"live","sources":[],"evidence":[],"error":True})
    try:
        result = live_report(message, request.history)
        result["conversation_id"] = request.conversation_id
        return result
    except Exception as exc:
        # Return a clear non-empty error, never a blank response or fake sources.
        return {
            "answer": "I can reach the Personal AI backend, but live web search failed. Please check the Render service logs and confirm outbound internet access is available.",
            "source": "live", "sources": [], "evidence": [], "error": True,
            "error_type": type(exc).__name__, "conversation_id": request.conversation_id,
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

    # Live/current-information questions use real web retrieval in Auto mode.
    # Ordinary study/programming questions continue through the local-first path.
    if mode == "auto" and is_live_query(message):
        try:
            result = live_report(message, request.history)
            result["conversation_id"] = request.conversation_id
            return result
        except Exception:
            return {
                "answer": "I can reach the Personal AI backend, but live web search failed. Please check the Render service logs and confirm outbound internet access is available.",
                "source": "live", "sources": [], "evidence": [], "error": True,
                "conversation_id": request.conversation_id,
            }

    # Math is a deterministic first pass across ALL modes. This prevents
    # calculator questions from entering fuzzy knowledge retrieval or Gemini.
    calculation = calculation_answer(message)
    if calculation is not None:
        return {
            "answer": calculation,
            "source": "calculator",
            "fallback": False,
            "calculation": True,
            "conversation_id": request.conversation_id,
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
                yield sse("done", {
                    "answer": "Gemini is currently unavailable. Check that at least one valid Gemini API key is configured in the Render service environment variables.",
                    "source": "error",
                    "fallback": False,
                    "conversation_id": request.conversation_id,
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
        for chunk, key_name in gemini_stream(message, history, answer):
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

        yield sse("done", {
            "answer": "I couldn't generate an answer because Gemini is unavailable and no suitable local answer was found. Check the Gemini API keys in the Render environment variables, then try again.",
            "source": "error",
            "fallback": False,
            "local_score": score,
            "local_topic": topic,
            "conversation_id": request.conversation_id,
        })

    return StreamingResponse(
        stream_auto(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
