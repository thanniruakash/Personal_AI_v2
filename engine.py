import os
import re
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from knowledge_loader import KnowledgeLoader
from search import KnowledgeSearch
from online_search import OnlineSearch
from live_data import LiveData
from ai_api import AIAPI
from calculator import Calculator


class SafeMath:
    """Safe arithmetic for +, -, *, /, %, //, ** and parentheses."""

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression):
        try:
            tree = ast.parse(expression, mode="eval")
            return cls._node(tree.body)
        except Exception:
            return None

    @classmethod
    def _node(cls, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in cls.OPERATORS:
            left = cls._node(node.left)
            right = cls._node(node.right)
            if left is None or right is None:
                raise ValueError("invalid")
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("large power")
            return cls.OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.OPERATORS:
            value = cls._node(node.operand)
            if value is None:
                raise ValueError("invalid")
            return cls.OPERATORS[type(node.op)](value)

        raise ValueError("unsupported")


class PersonalAI:

    def __init__(self):
        self.loader = KnowledgeLoader()
        self.search = KnowledgeSearch()
        self.online = OnlineSearch()
        self.live = LiveData()
        self.ai = AIAPI()
        self.calculator = Calculator()
        self.timezone = ZoneInfo("Asia/Kolkata")
        self.history = []

    # =========================================================
    # TENGlish / ABBREVIATION NORMALIZATION
    # =========================================================

    def normalize_question(self, question):
        q = str(question).strip()

        replacements = {
            r"\bevaru\b": "who",
            r"\benti\b": "what",
            r"\bemaindi\b": "what happened",
            r"\bela\b": "how",
            r"\benduku\b": "why",
            r"\beppudu\b": "when",
            r"\bekkada\b": "where",
            r"\bcheppu\b": "tell me",
            r"\bcheppandi\b": "tell me",
            r"\bivvu\b": "give me",
            r"\bivvandi\b": "give me",
            r"\bchupinchu\b": "show me",
            r"\bentha\b": "how much",
            r"\bemi\b": "what",
        }

        for pattern, replacement in replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        lower = q.lower()

        if re.search(r"\bcm\b", lower) and any(
            word in lower
            for word in (
                "minister", "chief", "state", "andhra", "ap",
                "telangana", "india", "who", "name"
            )
        ):
            q = re.sub(r"\bcm\b", "chief minister", q, flags=re.IGNORECASE)

        if re.search(r"\bpm\b", lower) and any(
            word in lower
            for word in (
                "minister", "prime", "india", "indian",
                "country", "who", "name"
            )
        ):
            q = re.sub(r"\bpm\b", "prime minister", q, flags=re.IGNORECASE)

        q = re.sub(
            r"\bdatabase management system\b", "DBMS", q, flags=re.IGNORECASE
        )
        q = re.sub(
            r"\bobject oriented programming\b", "OOP", q, flags=re.IGNORECASE
        )

        return q

    def now(self):
        return datetime.now(self.timezone)

    # =========================================================
    # EXPLICIT WEB MODE
    # =========================================================

    def is_web_request(self, msg):
        return msg.startswith((
            "web ", "web:", "online ", "online:",
            "search web ", "search online "
        ))

    def remove_web_prefix(self, msg):
        for prefix in (
            "search web ", "search online ",
            "web ", "web:", "online ", "online:"
        ):
            if msg.startswith(prefix):
                return msg[len(prefix):].strip()
        return msg

    # =========================================================
    # MATH
    # =========================================================

    def extract_math_expression(self, message):
        s = message.lower().strip()
        s = s.replace("×", "*").replace("÷", "/")
        s = s.replace(" plus ", "+")
        s = s.replace(" minus ", "-")
        s = s.replace(" multiplied by ", "*")
        s = s.replace(" times ", "*")
        s = s.replace(" divided by ", "/")
        s = s.replace(" remainder ", "%")

        candidate = re.sub(r"[^0-9+\-*/%().\s]", " ", s)
        candidate = re.sub(r"\s+", " ", candidate).strip()

        if not re.search(r"\d", candidate):
            return None
        if not re.search(r"[+\-*/%]", candidate):
            return None
        if re.fullmatch(r"\d{4}", candidate):
            return None

        return candidate

    def is_math_question(self, message):
        s = message.lower().strip()
        if self.extract_math_expression(s):
            return True
        return bool(re.search(r"\b(calculate|calculation|solve|what is)\b.*\d", s))

    def math_answer(self, message):
        expression = self.extract_math_expression(message)
        if not expression:
            return None

        result = SafeMath.evaluate(expression)

        if result is None:
            try:
                answer = self.calculator.ask(message)
                if answer:
                    return answer
            except Exception:
                pass
            return None

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"Answer: {result}"

    # =========================================================
    # TOPIC / LANGUAGE DETECTION
    # =========================================================

    TOPICS = {
        "docker": ("docker",),
        "kubernetes": ("kubernetes", "k8s"),
        "dbms": ("dbms", "database"),
        "dsa": ("dsa", "data structure", "algorithm"),
        "web": ("web", "website"),
    }

    def detect_language(self, msg):
        lower = msg.lower()

        for language in (
            "javascript", "typescript", "python", "java",
            "c++", "c#", "html", "css", "sql", "php"
        ):
            if re.search(rf"\b{re.escape(language)}\b", lower):
                return language

        if re.search(r"\bc language\b|\bc programming\b", lower):
            return "c"

        if re.search(
            r"\bc\s+(syntax|loop|loops|program|code|array|function)\b",
            lower
        ):
            return "c"

        return None

    def detect_topic(self, msg):
        language = self.detect_language(msg)
        if language:
            return language

        lower = msg.lower()
        for topic, words in self.TOPICS.items():
            if any(word in lower for word in words):
                return topic

        return None

    def is_coding_question(self, msg):
        lower = msg.lower()
        coding_words = (
            "code", "program", "programming", "debug", "debugging",
            "error", "exception", "algorithm", "function", "class",
            "method", "syntax", "compile", "compiler", "api",
            "script", "loop", "array", "write a program",
            "write code", "fix this code"
        )
        return (
            any(word in lower for word in coding_words)
            or self.detect_language(lower) is not None
        )

    def local_result_is_relevant(self, answer, question):
        if not answer:
            return False

        answer_text = str(answer).lower()
        language = self.detect_language(question)

        if language:
            if language == "c":
                return bool(re.search(
                    r"\bc\b|\bc language\b|\bc programming\b", answer_text
                ))
            return language in answer_text

        topic = self.detect_topic(question)

        if topic == "docker":
            return "docker" in answer_text

        if topic == "kubernetes":
            return "kubernetes" in answer_text or "k8s" in answer_text

        return True

    # =========================================================
    # GREETINGS
    # =========================================================

    def is_greeting(self, msg):
        return msg in {"hi", "hello", "hey", "hai", "hii", "helo"}

    # =========================================================
    # LOCAL KNOWLEDGE
    # =========================================================

    def search_local(self, question):
        try:
            answer = self.search.search(question)

            if answer and self.local_result_is_relevant(answer, question):
                return answer

            topic = self.detect_topic(question)

            if topic:
                focused = f"{topic} {question}"
                answer = self.search.search(focused)

                if answer and self.local_result_is_relevant(answer, question):
                    return answer

        except Exception:
            pass

        return None

    # =========================================================
    # MAIN ASK
    # =========================================================

    def ask(self, message):
        original = str(message).strip()

        if not original:
            return "Please enter a question."

        msg = original.lower().strip()
        self.history.append({"user": original})

        if self.is_greeting(msg):
            answer = "Hello! I'm Personal AI. How can I help you today?"
            self.history.append({"ai": answer})
            return answer

        if msg in {"bye", "goodbye", "exit", "quit"}:
            answer = "Goodbye! Have a great day."
            self.history.append({"ai": answer})
            return answer

        now = self.now()

        if "date" in msg and "update" not in msg:
            answer = now.strftime("Today's date is %d %B %Y.")
            self.history.append({"ai": answer})
            return answer

        if re.search(r"\btime\b", msg):
            answer = now.strftime("Current time is %I:%M %p (IST).")
            self.history.append({"ai": answer})
            return answer

        if (
            msg == "day"
            or "what day" in msg
            or "today's day" in msg
            or "today day" in msg
        ):
            answer = now.strftime("Today is %A.")
            self.history.append({"ai": answer})
            return answer

        if re.search(r"\bmonth\b", msg):
            answer = now.strftime("Current month is %B.")
            self.history.append({"ai": answer})
            return answer

        if re.search(r"\byear\b", msg):
            answer = now.strftime("Current year is %Y.")
            self.history.append({"ai": answer})
            return answer

        normalized = self.normalize_question(original)

        # WEB MODE = ONLINE ONLY. No local C/Java/Python fallback.
        if self.is_web_request(msg):
            query = self.remove_web_prefix(msg)

            try:
                live_answer = self.live.get_price(query)
                if live_answer:
                    self.history.append({"ai": live_answer})
                    return live_answer
            except Exception:
                pass

            try:
                online_answer = self.online.search(query)
                if online_answer:
                    self.history.append({"ai": online_answer})
                    return online_answer
            except Exception:
                pass

            return "I couldn't find a reliable online answer."

        # CALCULATOR
        if self.is_math_question(original):
            answer = self.math_answer(original)
            if answer:
                self.history.append({"ai": answer})
                return answer

        # LIVE DATA WITHOUT "web" PREFIX
        try:
            live_answer = self.live.get_price(normalized)
            if live_answer:
                self.history.append({"ai": live_answer})
                return live_answer
        except Exception:
            pass

        # LOCAL KNOWLEDGE FIRST
        local_answer = self.search_local(normalized)

        if local_answer:
            # Normal factual questions stay local.
            if not self.is_coding_question(msg):
                self.history.append({"ai": local_answer})
                return local_answer

            # Direct coding explanations stay local; generated code can use AI.
            if self._looks_like_direct_knowledge(local_answer, msg):
                self.history.append({"ai": local_answer})
                return local_answer

            try:
                ai_answer = self.ai.ask(normalized, local_answer)
                if ai_answer:
                    self.history.append({"ai": ai_answer})
                    return ai_answer
            except TypeError:
                try:
                    ai_answer = self.ai.ask(normalized)
                    if ai_answer:
                        self.history.append({"ai": ai_answer})
                        return ai_answer
                except Exception:
                    pass
            except Exception:
                pass

            self.history.append({"ai": local_answer})
            return local_answer

        # MODERN AI ONLY FOR CODING/PROGRAMMING QUESTIONS
        if self.is_coding_question(msg):
            try:
                ai_answer = self.ai.ask(normalized)
                if ai_answer:
                    self.history.append({"ai": ai_answer})
                    return ai_answer
            except Exception:
                pass

        # IMPORTANT: ordinary offline questions do NOT silently go online.
        return "I don't have a reliable answer for that yet."

    def _looks_like_direct_knowledge(self, answer, question):
        text = str(answer).lower()

        if "source:" in text or "wikipedia" in text:
            return True

        if any(
            phrase in question.lower()
            for phrase in ("what is", "what are", "define", "meaning", "explain", "difference")
        ):
            return True

        return False


if __name__ == "__main__":
    ai = PersonalAI()

    print("=== Personal AI v2 ===")
    print("Local knowledge first.")
    print("Live data for supported prices.")
    print("Use 'web <question>' for online-only search.")
    print("Math: + - * / %")
    print("Tenglish abbreviations are supported.")
    print("Type 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()

        if message.lower() in {"quit", "exit", "bye"}:
            print("AI: Goodbye! Have a great day.")
            break

        try:
            answer = ai.ask(message)
        except Exception as error:
            answer = f"I couldn't process that question: {error}"

        print("AI:", answer)
        
