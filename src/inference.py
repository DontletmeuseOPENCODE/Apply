from llama_cpp import Llama
import sys

MODEL_PATH = "outputs/ggml-model-q4_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_batch=128,
    verbose=False
)

def chat(prompt, system="You are a helpful coding and technical assistant."):
    formatted = f"<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    output = llm(
        formatted,
        max_tokens=256,
        stop=["<|end|>", "<|user|>"],
        echo=False
    )
    return output["choices"][0]["text"].strip()

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function to reverse a string."
    print("🤖", chat(query))
