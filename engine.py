import os
import json
import random

from datetime import datetime
from zoneinfo import ZoneInfo

from knowledge_loader import KnowledgeLoader
from search import KnowledgeSearch
from online_search import OnlineSearch


class PersonalAI:

    def __init__(self):

        # Existing knowledge loader.
        self.loader = KnowledgeLoader()

        # NEW strict knowledge search.
        self.search = KnowledgeSearch()

        # NEW online search.
        self.online = OnlineSearch()

        # IMPORTANT:
        # Colab normally uses UTC.
        # We want Indian Standard Time.
        self.timezone = ZoneInfo(
            "Asia/Kolkata"
        )

    # =========================================================
    # LEGACY INTENT SEARCH
    # =========================================================

    def search_intents(self, message):

        message_words = set(
            message.lower().split()
        )

        intents_folder = "intents"

        if not os.path.isdir(
            intents_folder
        ):
            return None

        best_response = None
        best_score = 0

        for filename in os.listdir(
            intents_folder
        ):

            if not filename.endswith(
                ".json"
            ):
                continue

            filepath = os.path.join(
                intents_folder,
                filename
            )

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

            except Exception:
                continue

            for intent in data.get(
                "intents",
                []
            ):

                for pattern in intent.get(
                    "patterns",
                    []
                ):

                    pattern_words = set(
                        pattern.lower().split()
                    )

                    useful_message = {
                        word
                        for word in message_words
                        if (
                            len(word) > 1
                            or word == "c"
                        )
                    }

                    useful_pattern = {
                        word
                        for word in pattern_words
                        if (
                            len(word) > 1
                            or word == "c"
                        )
                    }

                    if not useful_message:
                        continue

                    overlap = len(
                        useful_message &
                        useful_pattern
                    )

                    coverage = (
                        overlap /
                        len(useful_message)
                    )

                    if (
                        coverage >= 0.60
                        and overlap > best_score
                    ):

                        best_score = overlap

                        responses = intent.get(
                            "responses",
                            []
                        )

                        if responses:

                            best_response = (
                                random.choice(
                                    responses
                                )
                            )

        return best_response

    # =========================================================
    # CURRENT IST TIME
    # =========================================================

    def _now(self):

        return datetime.now(
            self.timezone
        )

    # =========================================================
    # WEB COMMAND DETECTION
    # =========================================================

    def _is_web_request(self, msg):

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

    def _remove_web_prefix(self, msg):

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
    # MAIN AI
    # =========================================================

    def ask(self, message):

        msg = str(
            message
        ).lower().strip()

        if not msg:

            return (
                "Please enter a question."
            )

        # =====================================================
        # GREETINGS
        # =====================================================

        if msg in {
            "hi",
            "hello",
            "hey",
            "hai"
        }:

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
        # IST DATE / TIME
        # =====================================================

        now = self._now()

        # DATE
        if (
            "date" in msg
            and "update" not in msg
        ):

            return now.strftime(
                "Today's date is %d %B %Y."
            )

        # TIME
        if "time" in msg:

            return now.strftime(
                "Current time is %I:%M %p (IST)."
            )

        # DAY
        if (
            msg == "day"
            or "today's day" in msg
            or "what day" in msg
        ):

            return now.strftime(
                "Today is %A."
            )

        # MONTH
        if "month" in msg:

            return now.strftime(
                "Current month is %B."
            )

        # YEAR
        if "year" in msg:

            return now.strftime(
                "Current year is %Y."
            )

        # =====================================================
        # EXPLICIT ONLINE MODE
        # =====================================================

        if self._is_web_request(
            msg
        ):

            web_query = (
                self._remove_web_prefix(
                    msg
                )
            )

            online_answer = (
                self.online.search(
                    web_query
                )
            )

            if online_answer:

                return online_answer

            return (
                "I couldn't find a reliable "
                "online answer right now."
            )

        # =====================================================
        # 1. SEARCH ALL LOCAL KNOWLEDGE
        # =====================================================

        local_answer = (
            self.search.search(
                message
            )
        )

        if local_answer:

            return local_answer

        # =====================================================
        # 2. ONLINE FALLBACK
        # =====================================================

        online_answer = (
            self.online.search(
                message
            )
        )

        if online_answer:

            return online_answer

        # =====================================================
        # 3. NEVER GUESS
        # =====================================================

        return (
            "I don't have a reliable answer "
            "for that yet. I checked the local "
            "knowledge base and online search."
        )


# =============================================================
# PROGRAM START
# =============================================================

if __name__ == "__main__":

    ai = PersonalAI()

    print(
        "=== Personal AI v2 ==="
    )

    print(
        "Offline knowledge is checked first."
    )

    print(
        "Use 'web <question>' for online-only search."
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
