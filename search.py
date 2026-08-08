import json
import re
from pathlib import Path


class KnowledgeSearch:
    """
    Strict local knowledge search.

    Searches every supported file recursively inside:
        knowledge/
        intents/

    New knowledge files are automatically detected.
    """

    SUPPORTED = {
        ".json",
        ".txt",
        ".md",
        ".csv",
        ".yaml",
        ".yml",
    }

    STOP_WORDS = {
        "a", "an", "the",
        "is", "are", "am", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "for", "from", "with",
        "and", "or", "but",
        "can", "could", "would", "should", "will",
        "do", "does", "did",
        "i", "me", "my", "you", "your",
        "we", "our", "they", "their", "it",
        "this", "that", "these", "those",
        "what", "why", "when", "where", "who", "which", "how",
        "tell", "give", "show", "explain", "about", "please",
        "define", "meaning",
    }

    def __init__(self, folder="knowledge"):
        self.folder = Path(folder)

        # Search both knowledge and intents.
        self.roots = [
            self.folder,
            Path("intents")
        ]

    # ---------------------------------------------------------
    # TOKENIZER
    # ---------------------------------------------------------

    def _tokens(self, text):
        text = str(text).lower()

        # Keeps programming terms such as:
        # c
        # c++
        # c#
        # dbms
        # oop
        raw = re.findall(
            r"[a-z0-9]+(?:\+\+|#)?",
            text
        )

        return {
            token
            for token in raw
            if token not in self.STOP_WORDS
        }

    # ---------------------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------------------

    def _clean(self, value):

        if value is None:
            return ""

        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                ensure_ascii=False
            )

        return str(value)

    # ---------------------------------------------------------
    # CREATE SEARCH CANDIDATE
    # ---------------------------------------------------------

    def _make_candidate(
        self,
        context,
        answer,
        source
    ):

        context = self._clean(context).strip()
        answer = self._clean(answer).strip()

        if not context or not answer:
            return None

        return {
            "text": context,
            "answer": answer,
            "source": source
        }

    # ---------------------------------------------------------
    # READ JSON
    # ---------------------------------------------------------

    def _extract_json_candidates(
        self,
        data,
        source,
        context=""
    ):

        candidates = []

        # -----------------------------
        # DICTIONARY
        # -----------------------------

        if isinstance(data, dict):

            # Handle intent JSON files.
            if isinstance(
                data.get("intents"),
                list
            ):

                for intent in data["intents"]:

                    if not isinstance(
                        intent,
                        dict
                    ):
                        continue

                    patterns = intent.get(
                        "patterns",
                        []
                    )

                    responses = intent.get(
                        "responses",
                        []
                    )

                    if not isinstance(
                        patterns,
                        list
                    ):
                        patterns = [patterns]

                    if not isinstance(
                        responses,
                        list
                    ):
                        responses = [responses]

                    for pattern in patterns:

                        for response in responses:

                            candidate = self._make_candidate(
                                f"{context} {pattern}",
                                response,
                                source
                            )

                            if candidate:
                                candidates.append(
                                    candidate
                                )

                return candidates

            # -----------------------------
            # NORMAL QUESTION / ANSWER JSON
            # -----------------------------

            question_keys = [
                "question",
                "query",
                "pattern",
                "title",
                "name",
                "topic"
            ]

            answer_keys = [
                "answer",
                "response",
                "content",
                "text",
                "description",
                "definition",
                "explanation",
                "solution",
                "result"
            ]

            question_parts = []
            answer_parts = []

            for key in question_keys:

                if key in data:

                    value = data[key]

                    if isinstance(
                        value,
                        (str, int, float)
                    ):
                        question_parts.append(
                            str(value)
                        )

            for key in answer_keys:

                if key in data:

                    value = data[key]

                    if isinstance(
                        value,
                        (str, int, float)
                    ):
                        answer_parts.append(
                            str(value)
                        )

            if question_parts and answer_parts:

                candidate = self._make_candidate(
                    f"{context} {' '.join(question_parts)}",
                    "\n".join(answer_parts),
                    source
                )

                if candidate:
                    candidates.append(
                        candidate
                    )

            # -----------------------------
            # SEARCH EVERY OTHER FIELD
            # -----------------------------

            for key, value in data.items():

                if key in question_keys:
                    continue

                if key in answer_keys:
                    continue

                new_context = (
                    f"{context} {key}"
                ).strip()

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool
                    )
                ):

                    candidate = self._make_candidate(
                        new_context,
                        value,
                        source
                    )

                    if candidate:
                        candidates.append(
                            candidate
                        )

                else:

                    candidates.extend(
                        self._extract_json_candidates(
                            value,
                            source,
                            new_context
                        )
                    )

        # -----------------------------
        # LIST
        # -----------------------------

        elif isinstance(data, list):

            for index, item in enumerate(data):

                candidates.extend(
                    self._extract_json_candidates(
                        item,
                        source,
                        f"{context} item{index}"
                    )
                )

        return candidates

    # ---------------------------------------------------------
    # LOAD ONE FILE
    # ---------------------------------------------------------

    def _load_file(self, path):

        try:

            # JSON
            if path.suffix.lower() == ".json":

                with path.open(
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                return self._extract_json_candidates(
                    data,
                    str(path),
                    path.stem
                )

            # TEXT / MARKDOWN / CSV / YAML
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                text = f.read()

            candidate = self._make_candidate(
                f"{path.stem} {text}",
                text,
                str(path)
            )

            if candidate:
                return [candidate]

            return []

        except Exception as error:

            print(
                f"[KnowledgeSearch] Skipped {path}: {error}"
            )

            return []

    # ---------------------------------------------------------
    # COLLECT EVERYTHING
    # ---------------------------------------------------------

    def _collect_candidates(self):

        candidates = []
        seen = set()

        for root in self.roots:

            if not root.exists():
                continue

            # RECURSIVE SEARCH
            for path in root.rglob("*"):

                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.SUPPORTED:
                    continue

                # Ignore Python cache / Git files.
                if any(
                    part in {
                        ".git",
                        "__pycache__",
                        "venv",
                        ".venv"
                    }
                    for part in path.parts
                ):
                    continue

                path_string = str(path)

                if path_string in seen:
                    continue

                seen.add(path_string)

                candidates.extend(
                    self._load_file(path)
                )

        return candidates

    # ---------------------------------------------------------
    # SCORE
    # ---------------------------------------------------------

    def _score(
        self,
        query,
        candidate
    ):

        query_tokens = self._tokens(
            query
        )

        if not query_tokens:
            return 0.0

        text = candidate["text"].lower()

        document_tokens = self._tokens(
            text
        )

        overlap = (
            query_tokens &
            document_tokens
        )

        if not overlap:
            return 0.0

        coverage = (
            len(overlap) /
            len(query_tokens)
        )

        score = coverage

        # Exact phrase bonus.
        normalized_query = (
            " ".join(
                str(query).lower().split()
            )
        )

        normalized_text = (
            " ".join(text.split())
        )

        if normalized_query in normalized_text:
            score += 1.0

        # All meaningful words bonus.
        if all(
            token in document_tokens
            for token in query_tokens
        ):
            score += 0.25

        # Small filename bonus.
        source_tokens = self._tokens(
            candidate["source"]
        )

        source_overlap = (
            query_tokens &
            source_tokens
        )

        score += min(
            len(source_overlap) * 0.05,
            0.15
        )

        return score

    # ---------------------------------------------------------
    # PUBLIC SEARCH
    # ---------------------------------------------------------

    def search(self, query):

        query = str(query).strip()

        if not query:
            return None

        candidates = (
            self._collect_candidates()
        )

        if not candidates:
            return None

        query_tokens = self._tokens(
            query
        )

        if not query_tokens:
            return None

        scored = []

        for candidate in candidates:

            score = self._score(
                query,
                candidate
            )

            if score <= 0:
                continue

            candidate_tokens = self._tokens(
                candidate["text"]
            )

            overlap = (
                query_tokens &
                candidate_tokens
            )

            coverage = (
                len(overlap) /
                len(query_tokens)
            )

            # STRICT MATCHING
            #
            # Example:
            #
            # "What is Docker?"
            #
            # Meaningful word = Docker
            #
            # Java answer does NOT contain Docker.
            #
            # Therefore Java cannot be returned.

            if (
                len(query_tokens) >= 2
                and coverage < 0.60
            ):
                continue

            scored.append(
                (
                    score,
                    coverage,
                    candidate
                )
            )

        if not scored:
            return None

        scored.sort(
            key=lambda item: (
                item[0],
                item[1]
            ),
            reverse=True
        )

        best_score, best_coverage, best = (
            scored[0]
        )

        # Final safety check.
        if (
            len(query_tokens) > 1
            and best_coverage < 0.60
        ):
            return None

        return best["answer"]
