import re


class KnowledgeSearch:
    def __init__(self, knowledge):
        self.knowledge = knowledge

    def search(self, query):
        query = query.lower()

        results = []

        for item in self.knowledge:
            content = str(item["content"]).lower()

            score = 0

            words = re.findall(r"\w+", query)

            for word in words:
                if word in content:
                    score += 1

            if score > 0:
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)

        return [item for score, item in results]


if __name__ == "__main__":
    from loader import KnowledgeLoader

    loader = KnowledgeLoader()
    knowledge = loader.load()

    search = KnowledgeSearch(knowledge)

    print("=" * 40)
    print("Personal AI Search")
    print("=" * 40)

    while True:
        question = input("\nSearch: ")

        if question.lower() in ["exit", "quit"]:
            break

        results = search.search(question)

        if not results:
            print("No results found.")
            continue

        print(f"\nFound {len(results)} results.\n")

        for item in results[:5]:
            print(item["path"])
