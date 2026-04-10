from memory_index import rebuild_index
from ai_brain import ask_ai
import sys

COMMANDS = {
    "/reload": lambda: (rebuild_index(), print("🔁 Memory index rebuilt.")),
    "/exit": lambda: (print("👋 Goodbye, see you next time!"), sys.exit(0)),
}


def main() -> None:
    print("🌀 AI Terminal — OpenAI SDK v1")
    print("Type a message or use: /reload /exit")
    while True:
        try:
            ui = input("You: ").strip()
            if not ui:
                continue
            if ui.startswith("/"):
                fn = COMMANDS.get(ui.split()[0])
                print("AI:" if fn is None else "", end="")
                if fn: fn()
                else: print("I don’t recognize that command.")            
            else:
                print("AI:", ask_ai(ui))
        except KeyboardInterrupt:
            print("\nbye"); break

if __name__ == "__main__":
    main()
