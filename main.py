"""SELF-RAG System – FastAPI Entry Point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.selfrag import router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

app = FastAPI(
    title="SELF-RAG System",
    description="Implements the SELF-RAG paper (Asai et al., 2023). LLM decides when to retrieve, evaluates passage relevance (IS_REL), checks factual support (IS_SUP), rates utility (IS_USE), and self-corrects through reflection rounds.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "SELF-RAG System",
        "version": "1.0.0",
        "paper": "SELF-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)",
        "arxiv": "https://arxiv.org/abs/2310.11511",
        "reflection_tokens": {
            "RETRIEVE": "Decides whether retrieval is needed",
            "IS_REL": "Evaluates passage relevance (0-1 score)",
            "IS_SUP": "Checks if response is factually supported",
            "IS_USE": "Rates overall utility (1-5 scale)",
        },
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
