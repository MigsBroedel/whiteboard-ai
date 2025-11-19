from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File
import requests
import os

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
Você é um assistente que converte falas humanas em anotações para um whiteboard digital.

Sempre responda APENAS em JSON válido.
"""

@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    temp_path = "/tmp/audio.mp3"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    segments, info = model.transcribe(temp_path, beam_size=5)
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
        }
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]
s