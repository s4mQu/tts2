import asyncio
import io
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from chatterbox.tts import ChatterboxTTS, Conditionals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VOICE_REF   = Path("sydney.wav")
CONDS_CACHE = Path("conditionals.pt")
DEVICE      = "cuda"
MAX_CHUNK_CHARS = 220

state: dict = {}

QUALITY_PRESETS = {
    "natural": {
        "exaggeration": 0.35,
        "cfg_weight": 0.72,
        "temperature": 0.58,
        "repetition_penalty": 1.2,
        "top_p": 0.92,
        "min_p": 0.1,
    },
    "balanced": {
        "exaggeration": 0.45,
        "cfg_weight": 0.62,
        "temperature": 0.64,
        "repetition_penalty": 1.2,
        "top_p": 0.93,
        "min_p": 0.09,
    },
    "expressive": {
        "exaggeration": 0.7,
        "cfg_weight": 0.45,
        "temperature": 0.78,
        "repetition_penalty": 1.15,
        "top_p": 0.95,
        "min_p": 0.07,
    },
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def split_long_text(text: str, max_chunk_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []

    def push_chunk(chunk: str) -> None:
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chunk_chars:
            push_chunk(sentence)
            continue

        parts = re.split(r"(?<=[,;:])\s+", sentence)
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chunk_chars:
                current = candidate
                continue

            if current:
                push_chunk(current)

            if len(part) <= max_chunk_chars:
                current = part
                continue

            words = part.split(" ")
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if len(candidate) <= max_chunk_chars:
                    current = candidate
                else:
                    push_chunk(current)
                    current = word

        if current:
            push_chunk(current)

    return chunks


def generate_with_chunks(
    model: ChatterboxTTS,
    chunks: list[str],
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    min_p: float,
    sentence_pause_ms: int,
) -> torch.Tensor:
    pause_samples = int((sentence_pause_ms / 1000) * model.sr)
    audio_parts: list[torch.Tensor] = []

    for idx, chunk in enumerate(chunks):
        wav = model.generate(
            chunk,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            min_p=min_p,
            audio_prompt_path=None,
        )
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        audio_parts.append(wav)

        if idx < len(chunks) - 1 and pause_samples > 0:
            silence = torch.zeros(
                (wav.shape[0], pause_samples),
                dtype=wav.dtype,
                device=wav.device,
            )
            audio_parts.append(silence)

    if not audio_parts:
        raise ValueError("No audio generated from text chunks")

    return torch.cat(audio_parts, dim=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading ChatterboxTTS onto GPU...")
    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    if CONDS_CACHE.exists():
        logging.info("Loading cached conditionals from %s", CONDS_CACHE)
        model.conds = Conditionals.load(CONDS_CACHE, map_location=DEVICE)
        model.conds = model.conds.to(DEVICE)
    else:
        logging.info("Computing conditionals from %s (one-time, will cache)...", VOICE_REF)
        model.prepare_conditionals(str(VOICE_REF), exaggeration=0.5)
        model.conds.save(CONDS_CACHE)
        logging.info("Conditionals saved to %s", CONDS_CACHE)

    state["model"] = model
    state["lock"]  = asyncio.Lock()
    logging.info("Server ready.")
    yield

    state.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="Chatterbox TTS Server", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    quality_preset: str = "balanced"
    sentence_pause_ms: int = Field(default=140, ge=0, le=500)
    exaggeration: float | None = Field(default=None, ge=0.1, le=1.5)
    cfg_weight: float | None = Field(default=None, ge=0.1, le=1.5)
    temperature: float | None = Field(default=None, ge=0.3, le=1.5)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.5, le=1.0)
    min_p: float | None = Field(default=None, ge=0.01, le=0.5)


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    preset = QUALITY_PRESETS.get(req.quality_preset)
    if not preset:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown quality_preset '{req.quality_preset}'. Valid options: {', '.join(sorted(QUALITY_PRESETS))}",
        )

    params = dict(preset)
    for key in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "top_p", "min_p"):
        value = getattr(req, key)
        if value is not None:
            params[key] = value

    cleaned_text = normalize_text(req.text)
    chunks = split_long_text(cleaned_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="text produced no valid chunks")

    model: ChatterboxTTS = state["model"]
    lock: asyncio.Lock   = state["lock"]

    async with lock:
        wav = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_with_chunks(
                model=model,
                chunks=chunks,
                exaggeration=params["exaggeration"],
                cfg_weight=params["cfg_weight"],
                temperature=params["temperature"],
                repetition_penalty=params["repetition_penalty"],
                top_p=params["top_p"],
                min_p=params["min_p"],
                sentence_pause_ms=req.sentence_pause_ms,
            ),
        )

    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
