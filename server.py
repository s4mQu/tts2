import asyncio
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from chatterbox.tts import ChatterboxTTS, Conditionals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VOICE_REF   = Path("female_shadowheart.flac")
CONDS_CACHE = Path("conditionals.pt")
DEVICE      = "cuda"

state: dict = {}


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
    exaggeration: float = 0.7
    cfg_weight: float = 0.3
    temperature: float = 0.8
    repetition_penalty: float = 1.2


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    model: ChatterboxTTS = state["model"]
    lock: asyncio.Lock   = state["lock"]

    async with lock:
        wav = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate(
                req.text,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                audio_prompt_path=None,
            ),
        )

    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
