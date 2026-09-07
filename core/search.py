
import re
import math
from collections import defaultdict, Counter
from difflib import SequenceMatcher


class KnowledgeSearch:

    # Common words which should not dominate retrieval.
    STOP_WORDS = {
        "what", "is", "a", "an", "the",
        "are", "was", "were", "who",
        "how", "why", "when", "where",
        "does", "do", "can", "could",
        "would", "should",
        "for", "to", "in", "on", "of",
        "from", "with", "by",
        "explain", "tell", "about",
        "please", "me",
        "difference", "between",
        "and", "or",
        "vs", "versus",
        "give", "define",
        "definition", "describe"
    }

    NORMALIZATION = {
        "variables": "variable",
        "functions": "function",
        "structures": "structure",
        "arrays": "array",
        "pointers": "pointer",
        "operators": "operator",
        "constants": "constant",
        "strings": "string",
        "loops": "loop",
        "classes": "class",
        "objects": "object",
        "methods": "method",
        "arguments": "argument",
        "parameters": "parameter",
        "programs": "program",
        "databases": "database",
        "networks": "network",
        "computers": "computer",
        "systems": "system",
        "applications": "application",
        "advantages": "advantage",
        "disadvantages": "disadvantage",
        "features": "feature",
        "types": "type",
        "processes": "process",
        "cycles": "cycle",
        "ecosystems": "ecosystem",
        "species": "species",
        "threats": "threat",
    }

    LANGUAGE_ALIASES = {
        "c": {"c", "c programming", "c language", "c programming language"},
        "python": {"python", "python programming"},
        "java": {"java", "java programming", "java language"},
        "javascript": {"javascript", "js"},
        "typescript": {"typescript", "ts"},
        "html": {"html"},
        "css": {"css"},
        "sql": {"sql"},
        "php": {"php"},
        "c++": {"c++", "cpp"},
        "c#": {"c#", "csharp"},
    }

    LANGUAGE_PATTERNS = [
        ("javascript", r"\bjavascript\b|\bjs\b"),
        ("typescript", r"\btypescript\b|\bts\b"),
        ("python", r"\bpython\b"),
        ("java", r"\bjava\b"),
        ("html", r"\bhtml\b"),
        ("css", r"\bcss\b"),
        ("sql", r"\bsql\b"),
        ("php", r"\bphp\b"),
        ("c++", r"\bc\+\+\b|\bcpp\b"),
        ("c#", r"\bc#\b|\bcsharp\b"),
    ]

    def __init__(self, knowledge):
        self.knowledge = knowledge

        self.topic_index = []
        self.word_index = defaultdict(set)
        self.topic_exact_index = defaultdict(set)

        # Document-frequency statistics for BM25-like scoring.
        self.document_frequency = Counter()
        self.total_documents = 0

        self._build_index()

    # ========================================================
    # TEXT PROCESSING
    # ========================================================

    def _words(self, text):
        return re.findall(
            r"[a-zA-Z0-9]+(?:\+\+|#)?",
            str(text).lower()
        )

    def _normalize_word(self, word):
        word = word.lower().strip()
        return self.NORMALIZATION.get(word, word)

    def _normalized_words(self, text):
        return [
            self._normalize_word(word)
            for word in self._words(text)
        ]

    def _clean_phrase(self, text):
        words = self._normalized_words(text)
        words = [
            w for w in words
            if w not in self.STOP_WORDS
        ]
        return " ".join(words)

    # ========================================================
    # QUERY UNDERSTANDING
    # ========================================================

    def _detect_language(self, query):
        q = str(query).lower()

        # Explicit language mentions.
        for language, pattern in self.LANGUAGE_PATTERNS:
            if re.search(pattern, q):
                return language

        # Standalone C requires special handling.
        if re.search(r"\bwhat\s+is\s+c\b", q):
            return "c"

        if re.search(
            r"\bin\s+c\b|"
            r"\bof\s+c\b|"
            r"\bc\s+programming\b|"
            r"\bc\s+language\b|"
            r"\bc\s+code\b",
            q
        ):
            return "c"

        return None

    def _query_type(self, query):
        q = str(query).lower().strip()

        if re.search(
            r"\bwhat\s+is\b|\bdefine\b|\bdefinition\b",
            q
        ):
            return "definition"

        if re.search(
            r"\bdifference\b|\bvs\b|\bversus\b|\bcompare\b",
            q
        ):
            return "comparison"

        if re.search(
            r"\bhow\s+does\b|\bhow\s+do\b|\bhow\s+to\b|\bsteps\b",
            q
        ):
            return "how"

        if re.search(
            r"\bexplain\b|\bdescribe\b|\bdetail\b",
            q
        ):
            return "explanation"

        return "normal"

    # ========================================================
    # INDEX BUILDING
    # ========================================================

    def _build_index(self):

        for item in self.knowledge:

            if not isinstance(item, dict):
                continue

            subject = str(item.get("subject", ""))
            section = str(item.get("section", ""))
            topics = item.get("topics", [])

            if not isinstance(topics, list):
                continue

            for topic in topics:

                if not isinstance(topic, dict):
                    continue

                record = {
                    "subject": subject,
                    "section": section,
                    "topic": str(topic.get("topic", "")),
                    "definition": str(topic.get("definition", "")),
                    "syntax": str(topic.get("syntax", "")),
                    "example": str(topic.get("example", "")),
                    "output": str(topic.get("output", "")),
                    "tips": topic.get("tips", [])
                }

                topic_id = len(self.topic_index)

                self.topic_index.append(record)

                # Full searchable text.
                text = self._text(record)

                words = set(self._normalized_words(text))

                # Document frequency.
                for word in words:
                    if word not in self.STOP_WORDS:
                        self.document_frequency[word] += 1

                # Word → documents.
                for word in words:
                    if word not in self.STOP_WORDS:
                        self.word_index[word].add(topic_id)

                # Exact topic phrase.
                topic_phrase = self._clean_phrase(record["topic"])

                if topic_phrase:
                    self.topic_exact_index[topic_phrase].add(topic_id)

        self.total_documents = len(self.topic_index)

    # ========================================================
    # DOCUMENT TEXT
    # ========================================================

    def _text(self, item):

        parts = [
            item.get("subject", ""),
            item.get("section", ""),
            item.get("topic", ""),
            item.get("definition", ""),
            item.get("syntax", ""),
            item.get("example", ""),
            item.get("output", "")
        ]

        tips = item.get("tips", [])

        if isinstance(tips, list):
            parts.extend(str(x) for x in tips)

        return " ".join(str(x) for x in parts)

    # ========================================================
    # FUZZY MATCHING
    # ========================================================

    def _similarity(self, a, b):

        a = str(a).lower().strip()
        b = str(b).lower().strip()

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        return SequenceMatcher(None, a, b).ratio()

    # ========================================================
    # BM25-LIKE SCORE
    # ========================================================

    def _idf(self, word):

        df = self.document_frequency.get(word, 0)

        if df <= 0:
            return 0.0

        return math.log(
            1.0 +
            (self.total_documents - df + 0.5) /
            (df + 0.5)
        )

    def _lexical_score(self, query_words, item):

        if not query_words:
            return 0.0

        fields = {
            "topic": self._normalized_words(item["topic"]),
            "section": self._normalized_words(item["section"]),
            "subject": self._normalized_words(item["subject"]),
            "definition": self._normalized_words(item["definition"]),
            "syntax": self._normalized_words(item["syntax"]),
            "example": self._normalized_words(item["example"]),
        }

        weights = {
            "topic": 6.0,
            "section": 2.5,
            "subject": 3.0,
            "definition": 1.8,
            "syntax": 1.0,
            "example": 0.7,
        }

        total = 0.0

        for field, words in fields.items():

            counts = Counter(words)

            for word in query_words:

                if word in self.STOP_WORDS:
                    continue

                tf = counts.get(word, 0)

                if tf:
                    total += (
                        self._idf(word)
                        * min(tf, 3)
                        * weights[field]
                    )

        return total

    # ========================================================
    # FIELD MATCHING
    # ========================================================

    def _field_score(self, query_phrase, item):

        topic = self._clean_phrase(item["topic"])
        section = self._clean_phrase(item["section"])
        subject = self._clean_phrase(item["subject"])

        score = 0.0

        # Exact topic.
        if query_phrase and query_phrase == topic:
            score += 7000

        # Query is contained in topic.
        elif query_phrase and query_phrase in topic:
            score += 5000

        # Topic contained in query.
        elif topic and topic in query_phrase:
            score += 4200

        # Fuzzy topic similarity.
        topic_sim = self._similarity(query_phrase, topic)

        if topic_sim >= 0.95:
            score += 4000
        elif topic_sim >= 0.88:
            score += 3000
        elif topic_sim >= 0.78:
            score += 1800
        elif topic_sim >= 0.68:
            score += 800

        # Section match.
        if query_phrase and query_phrase in section:
            score += 900

        # Subject match.
        if query_phrase and query_phrase == subject:
            score += 1600
        elif query_phrase and query_phrase in subject:
            score += 900

        return score

    # ========================================================
    # LANGUAGE RELEVANCE
    # ========================================================

    def _language_relevance(self, language, item):

        if not language:
            return 0

        text = self._text(item).lower()

        aliases = self.LANGUAGE_ALIASES.get(language, set())

        # Strong evidence anywhere in subject/section/topic.
        for alias in aliases:
            if re.search(
                r"(?<![a-z0-9])" +
                re.escape(alias) +
                r"(?![a-z0-9])",
                text
            ):
                return 1300

        return 0

    # ========================================================
    # DIRECT DEFINITION MATCH
    # ========================================================

    def _definition_score(self, query, query_phrase, item):

        if not re.search(
            r"\bwhat\s+is\b|\bdefine\b|\bdefinition\b",
            query.lower()
        ):
            return 0

        definition = item["definition"].lower().strip()
        topic = self._clean_phrase(item["topic"])

        score = 0

        # Definition directly starts by defining the topic.
        if topic:
            topic_pattern = re.escape(topic).replace(r"\ ", r"\s+")

            if re.search(
                rf"^(?:a|an|the)?\s*{topic_pattern}\s+(?:is|are)\b",
                definition
            ):
                score += 2500

        # Query concept occurs strongly in definition.
        qwords = set(query_phrase.split())

        if qwords:
            dwords = set(self._normalized_words(definition))
            overlap = len(qwords & dwords)

            score += overlap * 80

        return score

    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    def _retrieve_candidates(self, query_words):

        candidates = set()

        # Normal word retrieval.
        for word in query_words:
            candidates.update(
                self.word_index.get(word, set())
            )

        # Fuzzy topic retrieval for wording differences.
        # Only run against topic names, which is small and safe.
        if query_words:
            q = " ".join(query_words)

            for topic_id, item in enumerate(self.topic_index):

                topic = self._clean_phrase(item["topic"])

                if not topic:
                    continue

                sim = self._similarity(q, topic)

                if sim >= 0.62:
                    candidates.add(topic_id)

        return candidates

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(self, query):

        query = str(query).strip()

        if not query:
            return []

        language = self._detect_language(query)
        query_type = self._query_type(query)

        query_phrase = self._clean_phrase(query)

        query_words = [
            word
            for word in self._normalized_words(query)
            if word not in self.STOP_WORDS
        ]

        # Remove language words from concept scoring.
        if language:
            language_words = set()

            for alias in self.LANGUAGE_ALIASES.get(
                language, set()
            ):
                language_words.update(
                    self._normalized_words(alias)
                )

            query_words = [
                w for w in query_words
                if w not in language_words
            ]

        query_words = list(dict.fromkeys(query_words))

        # ----------------------------------------------------
        # SUBJECT-ONLY QUERY
        # Example: "What is Java?"
        # ----------------------------------------------------
        if language and not query_words:

            candidates = []

            for topic_id, item in enumerate(self.topic_index):

                subject = str(item["subject"]).lower()
                topic = str(item["topic"]).lower()
                section = str(item["section"]).lower()

                aliases = self.LANGUAGE_ALIASES.get(
                    language, set()
                )

                matched = False

                for alias in aliases:

                    if (
                        alias == subject.strip()
                        or alias in subject
                        or alias in topic
                        or alias in section
                    ):
                        matched = True
                        break

                if matched:
                    candidates.append(topic_id)

            # Prefer a topic explicitly defining the language.
            scored = []

            for topic_id in candidates:

                item = self.topic_index[topic_id]

                score = 1000

                text = self._text(item).lower()

                if re.search(
                    r"\bwhat\s+is\b",
                    item["topic"].lower()
                ):
                    score += 3500

                if language in text:
                    score += 500

                scored.append((score, topic_id))

            scored.sort(
                key=lambda x: (-x[0], self.topic_index[x[1]]["topic"].lower())
            )

            results = []

            for score, topic_id in scored[:20]:

                item = dict(self.topic_index[topic_id])
                item["content"] = self._text(item)
                item["_score"] = score

                results.append(item)

            return results

        # ----------------------------------------------------
        # GENERAL RETRIEVAL
        # ----------------------------------------------------
        candidates = self._retrieve_candidates(query_words)

        if not candidates:
            return []

        results = []

        for topic_id in candidates:

            item = self.topic_index[topic_id]

            lexical = self._lexical_score(
                query_words,
                item
            )

            field = self._field_score(
                query_phrase,
                item
            )

            definition = self._definition_score(
                query,
                query_phrase,
                item
            )

            language_score = self._language_relevance(
                language,
                item
            )

            # ------------------------------------------------
            # Phrase similarity against complete topic.
            # ------------------------------------------------
            topic_phrase = self._clean_phrase(
                item["topic"]
            )

            fuzzy = self._similarity(
                query_phrase,
                topic_phrase
            )

            fuzzy_score = fuzzy * 1800

            # ------------------------------------------------
            # Query word coverage.
            # ------------------------------------------------
            topic_words = set(
                self._normalized_words(item["topic"])
            )

            section_words = set(
                self._normalized_words(item["section"])
            )

            definition_words = set(
                self._normalized_words(item["definition"])
            )

            coverage = 0

            for word in query_words:

                if word in topic_words:
                    coverage += 450

                elif word in section_words:
                    coverage += 120

                elif word in definition_words:
                    coverage += 50

            # ------------------------------------------------
            # Strong topic overlap bonus.
            # ------------------------------------------------
            if query_words:

                topic_overlap = (
                    len(set(query_words) & topic_words)
                    / len(set(query_words))
                )

            else:
                topic_overlap = 0

            overlap_score = topic_overlap * 1800

            # ------------------------------------------------
            # Final score.
            # ------------------------------------------------
            score = (
                lexical * 35
                + field
                + definition
                + language_score
                + fuzzy_score
                + coverage
                + overlap_score
            )

            # Query-type adjustments.
            if query_type == "definition":

                if topic_overlap >= 0.8:
                    score += 900

            elif query_type == "comparison":

                if "difference" in query.lower():
                    score += 200

            elif query_type == "explanation":

                if topic_overlap >= 0.5:
                    score += 250

            result = dict(item)
            result["content"] = self._text(item)
            result["_score"] = round(score, 2)
            result["_fuzzy"] = round(fuzzy, 4)
            result["_topic_overlap"] = round(
                topic_overlap, 4
            )

            results.append(result)

        # ----------------------------------------------------
        # Sort.
        # ----------------------------------------------------
        results.sort(
            key=lambda x: (
                -x["_score"],
                -x["_topic_overlap"],
                -x["_fuzzy"],
                x["topic"].lower()
            )
        )

        if not results:
            return []

        # ----------------------------------------------------
        # CONFIDENCE GATE
        #
        # Prevent weak/unrelated local answers from being
        # treated as reliable answers.
        # ----------------------------------------------------
        top = results[0]

        top_score = top["_score"]

        second_score = (
            results[1]["_score"]
            if len(results) > 1
            else 0
        )

        margin = top_score - second_score

        # Strong direct topic match.
        strong_topic = (
            top["_topic_overlap"] >= 0.75
            and top["_fuzzy"] >= 0.72
        )

        # Very strong exact/near-exact match.
        very_strong = top_score >= 5000

        # Good result with meaningful separation.
        confident = (
            top_score >= 3000
            and (
                margin >= 250
                or top_score >= 4500
            )
        )

        # If the result is weak, return no result instead of
        # returning a random knowledge item.
        if not (
            strong_topic
            or very_strong
            or confident
        ):
            return []

        # Keep enough candidates for future reranking.
        return results[:20]
