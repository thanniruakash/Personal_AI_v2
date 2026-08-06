import json
from loader import KnowledgeLoader
from search import KnowledgeSearch


class PersonalAI:
    def __init__(self):
        self.loader = KnowledgeLoader()
        self.knowledge = self.loader.load()
        self.search = KnowledgeSearch()

    def ask(self, question):
        result = self.search.search(question)

        if result is None:
            return "Sorry, I couldn't find an answer."

        data = result["data"]

        if isinstance(data, dict):
            lines = []
            title = data.get("title") or data.get("topic") or "Knowledge"
            lines.append(f"=== {title} ===")

            for key, value in data.items():
                if key.lower() in ("title", "topic"):
                    continue
                lines.append(f"\n{key.capitalize()}:")
                if isinstance(value, list):
                    for item in value:
                        lines.append(f"• {item}")
                else:
                    lines.append(str(value))
            return "\n".join(lines)

        return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ai = PersonalAI()

    print("=" * 50)
    print("Personal AI v1")
    print("=" * 50)

    while True:
        q = input("\nYou: ")

        if q.lower() in ("exit", "quit"):
            break

        print("\nAI:")
        print(ai.ask(q))
