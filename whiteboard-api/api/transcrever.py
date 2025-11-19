from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File
import requests
import os

app = FastAPI()

model = WhisperModel("base", device="cpu", compute_type="int8")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
Você é um assistente que converte falas humanas em anotações para um whiteboard digital.

Sempre responda APENAS em JSON válido, seguindo exatamente esta estrutura:

{
  "notes": [
    {
      "id": "string",
      "text": "string",
      "x": number,
      "y": number,
      "width": number,
      "height": number
    }
  ],
  "connections": [
    {
      "from": "string",
      "to": "string"
    }
  ]
}

Regras importantes:

1. Nunca escreva nada fora do JSON.
2. Cada nota deve ser curta, clara e baseada nas ideias principais do texto.
3. Gere posições (x e y) entre 50 e 600.
4. width e height devem ser valores simples entre 150 e 300.
5. "id" deve ser único para cada nota.
6. "connections" deve relacionar conceitos que fazem sentido.
7. Se o texto for muito longo, divida em várias notas curtas.
8. Não explique o JSON. Apenas retorne o JSON.
9. Não use markdown.
"""

@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    # salvar temporariamente
    temp_path = "/tmp/audio.mp3"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # transcrever
    segments, info = model.transcribe(temp_path, beam_size=5)
    transcricao = " ".join([s.text for s in segments])

    # enviar para openrouter
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
