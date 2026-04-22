import sys
import requests

SERVER = "http://127.0.0.1:8000"
OUTPUT_PATH = "output.wav"


def synthesize(text: str, quality_preset: str = "expressive"):
    print(f"Synthesizing: {text!r}")
    resp = requests.post(
        f"{SERVER}/synthesize",
        json={"text": text, "quality_preset": quality_preset},
        timeout=120,
    )
    resp.raise_for_status()
    with open(OUTPUT_PATH, "wb") as f:
        f.write(resp.content)
    print(f"Saved {len(resp.content) / 1024:.1f} KB -> {OUTPUT_PATH}")


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg.strip()]
    preset = "natural"

    if len(args) >= 2 and args[0] in {"natural", "balanced", "expressive"}:
        preset = args[0]
        args = args[1:]

    text = " ".join(args) or "Hello, this is a test of the Chatterbox TTS server."
    synthesize(text, quality_preset=preset)
