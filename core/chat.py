from engine import PersonalAI
from memory import Memory


def main():
    ai = PersonalAI()
    memory = Memory()

    print("=" * 50)
    print("🤖 Personal AI v1")
    print("=" * 50)
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("\n👋 Goodbye!")
            break

        answer = ai.ask(question)

        print("\nAI:")
        print(answer)

        history = memory.get("history", [])
        history.append({
            "user": question,
            "ai": answer
        })

        memory.set("history", history)


if __name__ == "__main__":
    main()
