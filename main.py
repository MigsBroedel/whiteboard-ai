import asyncio
import os
import logging
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Usa modelo TINY para ser 4x mais rápido que base
model = WhisperModel("tiny", device="cpu", compute_type="int8")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """Você é um assistente que converte falas em anotações estruturadas para whiteboard.

Analise o texto e crie notas organizadas com posicionamento inteligente.

Responda APENAS com JSON válido neste formato:
{
  "notes": [
    {"text": "conteúdo da nota", "x": 50, "y": 50},
    {"text": "outra nota", "x": 250, "y": 50}
  ],
  "connections": []
}

Regras:
- Cada nota deve ter no máximo 80 caracteres
- Distribua as notas em uma grade (espaçamento: 220px horizontal, 120px vertical)
- Agrupe notas relacionadas próximas umas das outras
- Comece em x=50, y=50"""


async def run_transcription(path: str):
    """Executa transcrição em thread separada para não bloquear"""
    loop = asyncio.get_event_loop()
    
    def transcribe():
        try:
            segments, info = model.transcribe(path, language="pt")
            return " ".join([s.text for s in segments])
        except Exception as e:
            logger.error(f"Erro na transcrição: {e}")
            raise
    
    return await loop.run_in_executor(None, transcribe)


def get_llm_response(transcricao: str) -> Dict:
    """Chama LLM e retorna JSON estruturado"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",  # Modelo mais rápido e melhor
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcricao}
                ],
                "temperature": 0.3
            },
            timeout=20
        )
        
        if response.status_code != 200:
            logger.error(f"Erro LLM: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Erro ao processar com LLM")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Remove markdown se houver
        content = content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao parsear JSON: {e}")
        raise HTTPException(status_code=500, detail="Resposta inválida do LLM")
    except Exception as e:
        logger.error(f"Erro na chamada LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    """Endpoint de health check"""
    return {"status": "ok", "model": "tiny"}


@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    """
    Endpoint principal: recebe áudio, transcreve e organiza em notas
    """
    temp_path = "/tmp/audio.webm"
    
    try:
        # Salva arquivo temporário
        logger.info(f"Recebendo arquivo: {file.filename}")
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Transcrição (~ 2-5 segundos com modelo tiny)
        logger.info("Iniciando transcrição...")
        transcricao = await run_transcription(temp_path)
        logger.info(f"Transcrição concluída: {transcricao[:100]}...")
        
        if not transcricao or len(transcricao.strip()) < 5:
            return JSONResponse({
                "notes": [{"text": "Nenhuma fala detectada", "x": 50, "y": 50}],
                "connections": []
            })
        
        # Processamento LLM (~ 2-3 segundos)
        logger.info("Processando com LLM...")
        resultado = get_llm_response(transcricao)
        logger.info("Processamento concluído!")
        
        return JSONResponse(resultado)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro geral: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    finally:
        # Limpa arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)