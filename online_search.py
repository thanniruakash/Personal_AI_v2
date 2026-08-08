import json
import urllib.parse
import urllib.request


class OnlineSearch:

    """
    Online fallback.

    Uses public Wikipedia and DuckDuckGo endpoints.
    No API key is required.
    """

    def _get_json(self, url):

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PersonalAI-v2/1.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=8
        ) as response:

            return json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(self, query):

        query = str(query).strip()

        if not query:
            return None

        # =====================================================
        # 1. WIKIPEDIA
        # =====================================================

        try:

            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": 1
            })

            data = self._get_json(
                "https://en.wikipedia.org/w/api.php?"
                + params
            )

            results = (
                data
                .get("query", {})
                .get("search", [])
            )

            if results:

                title = results[0].get(
                    "title",
                    ""
                )

                summary_params = (
                    urllib.parse.urlencode({
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "redirects": 1,
                        "titles": title,
                        "format": "json",
                        "formatversion": 2
                    })
                )

                summary_data = self._get_json(
                    "https://en.wikipedia.org/w/api.php?"
                    + summary_params
                )

                pages = (
                    summary_data
                    .get("query", {})
                    .get("pages", [])
                )

                if pages:

                    extract = (
                        pages[0]
                        .get("extract", "")
                        .strip()
                    )

                    if extract:

                        return (
                            f"{extract}\n\n"
                            f"Source: Wikipedia ({title})"
                        )

        except Exception:
            pass

        # =====================================================
        # 2. DUCKDUCKGO
        # =====================================================

        try:

            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            })

            data = self._get_json(
                "https://api.duckduckgo.com/?"
                + params
            )

            abstract = (
                data
                .get("AbstractText", "")
                .strip()
            )

            if abstract:

                heading = data.get(
                    "Heading",
                    ""
                )

                if heading:

                    return (
                        f"{abstract}\n\n"
                        f"Source: DuckDuckGo ({heading})"
                    )

                return (
                    f"{abstract}\n\n"
                    "Source: DuckDuckGo"
                )

            # Related topic fallback.
            for item in data.get(
                "RelatedTopics",
                []
            ):

                if (
                    isinstance(item, dict)
                    and item.get("Text")
                ):

                    return (
                        item["Text"]
                        + "\n\n"
                        "Source: DuckDuckGo"
                    )

        except Exception:
            pass

        return None
