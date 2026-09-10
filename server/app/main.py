"""API FastAPI que expone el chat del RAG sobre Mondial Europe."""
from fastapi import FastAPI
from pydantic import BaseModel

from .agent import answer, llm_model, llm_provider

app = FastAPI(title="RAG Mondial Europe")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


# Chequeo simple de salud del servicio.
@app.get("/health")
def health() -> dict:
    """Devuelve el estado del servicio."""
    return {"status": "ok", "llm": llm_provider(), "model": llm_model()}


# Endpoint principal del chat: delega en el agente RAG.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Recibe un mensaje del usuario y devuelve la respuesta del agente."""
    return ChatResponse(answer=answer(req.message))
