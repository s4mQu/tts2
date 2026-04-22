import sys
import requests

SERVER = "http://127.0.0.1:8000"


def synthesize(text: str, output_path: str = "output.wav"):
    print(f"Synthesizing: {text!r}")
    resp = requests.post(f"{SERVER}/synthesize", json={"text": text}, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved {len(resp.content) / 1024:.1f} KB -> {output_path}")


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Hello, this is a test of the Chatterbox TTS server."
    synthesize(text)
