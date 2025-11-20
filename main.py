import asyncio
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File
import requests
import os
import logging
logging.basicConfig(level=logging.INFO)


app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
Você é um assistente que converte falas humanas em anotações para um whiteboard digital.
Sempre responda APENAS em JSON válido.
"""

async def run_transcription(path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: list(model.transcribe(path)))

@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    temp_path = "/tmp/audio.webm"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    segments, info = await run_transcription(temp_path)
    transcricao = " ".join([s.text for s in segments])

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": "openai/gpt-oss-20b:free",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcricao}
            ]
        },
        timeout=30  # <- muito importante!
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]
