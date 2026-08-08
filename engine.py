import os
import json
import random
import re

from datetime import datetime
from zoneinfo import ZoneInfo

from knowledge_loader import KnowledgeLoader
from search import KnowledgeSearch
from online_search import OnlineSearch
from live_data import LiveData
from ai_api import AIAPI


class PersonalAI:

    def __init__(self):

        self.loader = KnowledgeLoader()

        self.search = KnowledgeSearch()

        self.online = OnlineSearch()

        self.live = LiveData()

        self.ai = AIAPI()

        self.timezone = ZoneInfo(
            "Asia/Kolkata"
        )

    # =========================================================
    # NORMALIZE COMMON ABBREVIATIONS
    # =========================================================

    def normalize_question(self, question):

        q = question.strip()

        lower = q.lower()

        # ---------------------------------------------
        # Chief Minister
        # ---------------------------------------------

        if (
            re.search(
                r"\bcm\b",
                lower
            )
            and any(
                word in lower
                for word in [
                    "minister",
                    "chief",
                    "state",
                    "andhra",
                    "ap",
                    "telangana",
                    "india"
                ]
            )
        ):

            q = re.sub(
                r"\bcm\b",
                "chief minister",
                q,
                flags=re.IGNORECASE
            )

        # ---------------------------------------------
        # Prime Minister
        # ---------------------------------------------

        if (
            re.search(
                r"\bpm\b",
                lower
            )
            and any(
                word in lower
                for word in [
                    "minister",
                    "prime",
                    "india",
                    "country"
                ]
            )
        ):

            q = re.sub(
                r"\bpm\b",
                "prime minister",
                q,
                flags=re.IGNORECASE
            )

        # ---------------------------------------------
        # DBMS
        # ---------------------------------------------

        q = re.sub(
            r"\bdatabase management system\b",
            "DBMS",
            q,
            flags=re.IGNORECASE
        )

        # ---------------------------------------------
        # OOP
        # ---------------------------------------------

        q = re.sub(
            r"\bobject oriented programming\b",
            "OOP",
            q,
            flags=re.IGNORECASE
        )

        return q

    # =========================================================
    # IST
    # =========================================================

    def now(self):

        return datetime.now(
            self.timezone
        )

    # =========================================================
    # WEB REQUEST
    # =========================================================

    def is_web_request(self, msg):

        prefixes = (
            "web ",
            "web:",
            "online ",
            "online:",
            "search web ",
            "search online "
        )

        return msg.startswith(
            prefixes
        )

    # =========================================================
    # REMOVE WEB PREFIX
    # =========================================================

    def remove_web_prefix(self, msg):

        prefixes = (
            "search web ",
            "search online ",
            "web ",
            "web:",
            "online ",
            "online:"
        )

        for prefix in prefixes:

            if msg.startswith(
                prefix
            ):

                return msg[
                    len(prefix):
                ].strip()

        return msg

    # =========================================================
    # CODING QUESTION DETECTION
    # =========================================================

    def is_coding_question(self, msg):

        coding_words = [
            "code",
            "program",
            "programming",
            "debug",
            "debugging",
            "error",
            "exception",
            "algorithm",
            "function",
            "class",
            "method",
            "syntax",
            "compile",
            "compiler",
            "api",
            "script",
            "write a program",
            "write code",
            "fix this code"
        ]

        languages = [
            "python",
            "java",
            "javascript",
            "typescript",
            "c",
            "c++",
            "c#",
            "html",
            "css",
            "sql",
            "php"
        ]

        has_coding_word = any(
            word in msg
            for word in coding_words
        )

        has_language = any(
            re.search(
                rf"\b{re.escape(language)}\b",
                msg
            )
            for language in languages
        )

        return (
            has_coding_word
            or (
                has_language
                and any(
                    word in msg
                    for word in [
                        "example",
                        "how",
                        "write",
                        "create",
                        "fix"
                    ]
                )
            )
        )

    # =========================================================
    # GREETINGS
    # =========================================================

    def is_greeting(self, msg):

        return msg in {
            "hi",
            "hello",
            "hey",
            "hai"
        }

    # =========================================================
    # MAIN ASK
    # =========================================================

    def ask(self, message):

        original = str(
            message
        ).strip()

        if not original:

            return (
                "Please enter a question."
            )

        msg = original.lower().strip()

        # =====================================================
        # GREETING
        # =====================================================

        if self.is_greeting(msg):

            return (
                "Hello! I'm Personal AI. "
                "How can I help you today?"
            )

        # =====================================================
        # GOODBYE
        # =====================================================

        if msg in {
            "bye",
            "goodbye",
            "exit",
            "quit"
        }:

            return (
                "Goodbye! Have a great day."
            )

        # =====================================================
        # DATE / TIME
        # =====================================================

        now = self.now()

        if (
            "date" in msg
            and "update" not in msg
        ):

            return now.strftime(
                "Today's date is %d %B %Y."
            )

        if "time" in msg:

            return now.strftime(
                "Current time is %I:%M %p (IST)."
            )

        if (
            msg == "day"
            or "what day" in msg
            or "today's day" in msg
        ):

            return now.strftime(
                "Today is %A."
            )

        if "month" in msg:

            return now.strftime(
                "Current month is %B."
            )

        if "year" in msg:

            return now.strftime(
                "Current year is %Y."
            )

        # =====================================================
        # NORMALIZE
        # =====================================================

        normalized = (
            self.normalize_question(
                original
            )
        )

        # =====================================================
        # EXPLICIT WEB MODE
        # =====================================================

        if self.is_web_request(msg):

            web_query = (
                self.remove_web_prefix(
                    msg
                )
            )

            # Live data first.
            live_answer = (
                self.live.get_price(
                    web_query
                )
            )

            if live_answer:

                return live_answer

            # Online search.
            online_answer = (
                self.online.search(
                    web_query
                )
            )

            if online_answer:

                return online_answer

            # AI fallback.
            ai_answer = (
                self.ai.ask(
                    web_query
                )
            )

            if ai_answer:

                return ai_answer

            return (
                "I couldn't find a reliable "
                "online answer."
            )

        # =====================================================
        # LIVE DATA
        # =====================================================

        live_answer = (
            self.live.get_price(
                normalized
            )
        )

        if live_answer:

            return live_answer

        # =====================================================
        # LOCAL KNOWLEDGE
        # =====================================================

        local_answer = (
            self.search.search(
                normalized
            )
        )

        if local_answer:

            # Coding questions that need modern
            # generation can be improved by AI.
            if self.is_coding_question(
                msg
            ):

                ai_answer = (
                    self.ai.ask(
                        normalized,
                        local_answer
                    )
                )

                if ai_answer:

                    return ai_answer

            return local_answer

        # =====================================================
        # CODING / MODERN AI
        # =====================================================

        if self.is_coding_question(
            msg
        ):

            ai_answer = (
                self.ai.ask(
                    normalized
                )
            )

            if ai_answer:

                return ai_answer

        # =====================================================
        # ONLINE SEARCH
        # =====================================================

        online_answer = (
            self.online.search(
                normalized
            )
        )

        if online_answer:

            return online_answer

        # =====================================================
        # AI FALLBACK
        # =====================================================

        ai_answer = (
            self.ai.ask(
                normalized
            )
        )

        if ai_answer:

            return ai_answer

        # =====================================================
        # NEVER GUESS
        # =====================================================

        return (
            "I don't have a reliable answer "
            "for that yet."
        )


# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    ai = PersonalAI()

    print(
        "=== Personal AI v2 ==="
    )

    print(
        "Local knowledge + Live APIs + Online Search + AI"
    )

    print(
        "Use 'web <question>' for online-only mode."
    )

    print(
        "Type 'quit' to exit.\n"
    )

    while True:

        message = input(
            "You: "
        ).strip()

        if message.lower() in {
            "quit",
            "exit",
            "bye"
        }:

            print(
                "AI: Goodbye! Have a great day."
            )

            break

        print(
            "AI:",
            ai.ask(message)
            )
