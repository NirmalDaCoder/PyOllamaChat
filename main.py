from ollama import chat
import os.path


def load_history_file(file: str) -> list:
    if os.path.isfile(file):
        with open(file, "r") as history_file:
            return eval(history_file.read())
    else:
        return [{"role": "system", "content": "You are a useful assistant"}]


def ollama_chat(model: str, messages: list) -> list:
    print()
    user_input = input("User: ")
    print()
    response = ""
    response_stream = chat(
        model,
        messages=messages
        + [
            {"role": "user", "content": user_input},
        ],
        stream=True,
    )
    print("Assistant: ", end="")
    for response_part in response_stream:
        response += str(response_part.message.content)
        print(str(response_part.message.content), end="", flush=True)
    print()
    messages += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ]
    return messages


def main():
    messages = load_history_file("history.txt")
    try:
        while True:
            messages = ollama_chat("gemma3", messages)
    except KeyboardInterrupt:
        print()
        print("Exiting...")
    finally:
        hist = open("history.txt", "w")
        hist.write(str(messages))
        hist.close()


if __name__ == "__main__":
    main()
