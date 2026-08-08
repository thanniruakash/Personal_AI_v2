import os


class AIAPI:

    def __init__(self):

        self.client = None

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            return

        try:

            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key
            )

        except Exception as error:

            print(
                "[AIAPI] OpenAI initialization error:",
                error
            )

    # =========================================================
    # AVAILABLE?
    # =========================================================

    def available(self):

        return self.client is not None

    # =========================================================
    # ASK AI
    # =========================================================

    def ask(
        self,
        question,
        local_context=""
    ):

        if not self.available():

            return None

        system_prompt = """
You are the online intelligence layer of Personal AI v2.

Rules:

1. Answer the user's actual question.
2. Never change the subject.
3. Never answer C questions with Java information.
4. Never answer Java questions with Python information.
5. Respect explicit programming language names.
6. If the user asks for code, provide correct runnable code.
7. Explain code simply when useful.
8. Do not invent facts.
9. If information is uncertain or current, say so.
10. If local knowledge is provided, use it as context,
    but correct obvious outdated information when necessary.
11. Keep the answer focused on the question.
"""

        if local_context:

            system_prompt += f"""

Relevant local knowledge:

{local_context}
"""

        try:

            response = self.client.responses.create(
                model="gpt-5-mini",
                instructions=system_prompt,
                input=question
            )

            return response.output_text.strip()

        except Exception as error:

            print(
                "[AIAPI] API error:",
                error
            )

            return None
