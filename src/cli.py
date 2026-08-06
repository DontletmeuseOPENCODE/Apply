import argparse
from inference import chat

def main():
    parser = argparse.ArgumentParser(description="CodeChat Agent - CPU-optimized LLM assistant")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the model")
    parser.add_argument("-s", "--system", default="You are a helpful coding and technical assistant.")
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else "Hello! Who are you?"
    print("\n🤖", chat(prompt, args.system))

if __name__ == "__main__":
    main()
