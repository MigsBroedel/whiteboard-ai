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
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo TINY para velocidade máxima
model = WhisperModel("tiny", device="cpu", compute_type="int8")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """Você é um assistente que converte falas em anotações para whiteboard canvas.

Analise o texto transcrito e crie notas organizadas e posicionadas de forma inteligente.

IMPORTANTE: Responda APENAS com JSON válido, SEM markdown, SEM comentários, SEM texto adicional.

Formato EXATO do JSON:
{
  "notes": [
    {"text": "primeira nota aqui", "x": 100, "y": 100},
    {"text": "segunda nota aqui", "x": 350, "y": 100}
  ],
  "connections": []
}

REGRAS:
1. Cada nota deve ter no máximo 80 caracteres
2. Divida ideias longas em múltiplas notas
3. Use coordenadas em grade:
   - Primeira nota: x=100, y=100
   - Notas seguintes: x aumenta +250px (horizontal)
   - Nova linha: y aumenta +150px, x volta para 100
4. Agrupe notas relacionadas próximas
5. Máximo de 3 notas por linha
6. Mantenha espaçamento consistente

Exemplo para "precisamos comprar leite, ovos e pão amanhã":
{
  "notes": [
    {"text": "Lista de Compras", "x": 100, "y": 100},
    {"text": "🥛 Leite", "x": 100, "y": 250},
    {"text": "🥚 Ovos", "x": 350, "y": 250},
    {"text": "🍞 Pão", "x": 600, "y": 250},
    {"text": "📅 Comprar amanhã", "x": 100, "y": 400}
  ],
  "connections": []
}"""


async def run_transcription(path: str):
    """Executa transcrição em thread separada"""
    loop = asyncio.get_event_loop()
    
    def transcribe():
        try:
            segments, info = model.transcribe(path, language="pt")
            text = " ".join([s.text for s in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Erro na transcrição: {e}")
            raise
    
    return await loop.run_in_executor(None, transcribe)


def clean_json_response(content: str) -> str:
    """Remove markdown e limpa resposta JSON"""
    content = content.strip()
    
    # Remove blocos de código markdown
    if "```json" in content:
        content = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if content:
            content = content.group(1)
    elif "```" in content:
        content = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
        if content:
            content = content.group(1)
    
    # Remove texto antes e depois do JSON
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    
    return content.strip()


def get_llm_response(transcricao: str) -> Dict:
    """Chama LLM e retorna JSON estruturado para canvas"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transcrição: {transcricao}"}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            },
            timeout=25
        )
        
        if response.status_code != 200:
            logger.error(f"Erro LLM: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Erro ao processar com LLM")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Limpa resposta
        content = clean_json_response(content)
        logger.info(f"JSON limpo: {content[:200]}...")
        
        # Parseia JSON
        result = json.loads(content)
        
        # Valida estrutura
        if "notes" not in result:
            raise ValueError("JSON não contém campo 'notes'")
        
        if not isinstance(result["notes"], list):
            raise ValueError("Campo 'notes' deve ser uma lista")
        
        # Garante que todas as notas têm x, y e text
        for note in result["notes"]:
            if not all(k in note for k in ["text", "x", "y"]):
                raise ValueError("Cada nota deve ter 'text', 'x' e 'y'")
        
        # Garante campo connections
        if "connections" not in result:
            result["connections"] = []
        
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao parsear JSON: {e}\nConteúdo: {content}")
        # Retorna nota de erro como fallback
        return {
            "notes": [
                {
                    "text": "Erro: resposta do LLM inválida. Tente novamente.",
                    "x": 100,
                    "y": 100
                }
            ],
            "connections": []
        }
    except Exception as e:
        logger.error(f"Erro na chamada LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "whisper-tiny",
        "llm": "gpt-4o-mini"
    }


@app.post("/transcrever")
async def transcrever_audio(file: UploadFile = File(...)):
    """
    Endpoint principal: recebe áudio, transcreve e retorna notas para canvas
    """
    temp_path = "/tmp/audio.webm"
    
    try:
        # Salva arquivo
        logger.info(f"📥 Recebendo arquivo: {file.filename}")
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Transcrição (2-5s)
        logger.info("🎙️ Iniciando transcrição...")
        transcricao = await run_transcription(temp_path)
        logger.info(f"✅ Transcrição: '{transcricao[:150]}...'")
        
        if not transcricao or len(transcricao) < 3:
            return JSONResponse({
                "notes": [
                    {
                        "text": "⚠️ Nenhuma fala detectada no áudio",
                        "x": 100,
                        "y": 100
                    }
                ],
                "connections": []
            })
        
        # Processamento LLM (2-4s)
        logger.info("🤖 Processando com LLM...")
        resultado = get_llm_response(transcricao)
        logger.info(f"✅ {len(resultado['notes'])} notas criadas!")
        
        return JSONResponse(resultado)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro geral: {e}", exc_info=True)
        return JSONResponse({
            "notes": [
                {
                    "text": f"Erro: {str(e)[:100]}",
                    "x": 100,
                    "y": 100
                }
            ],
            "connections": []
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)