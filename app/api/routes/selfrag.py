"""SELF-RAG System – API routes."""
import shutil, tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List
from app.services.selfrag_service import SELFRAGService, get_selfrag_service

router = APIRouter(prefix="/selfrag", tags=["SELF-RAG"])

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
async def query(req: QueryRequest, svc: SELFRAGService = Depends(get_selfrag_service)):
    if len(req.question.strip()) < 5:
        raise HTTPException(400, "Question too short")
    return svc.query(req.question)

@router.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), svc: SELFRAGService = Depends(get_selfrag_service)):
    tmp = tempfile.mkdtemp()
    try:
        paths = []
        for f in files:
            dest = Path(tmp) / f.filename
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append(str(dest))
        return svc.ingest(paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@router.get("/health")
async def health():
    return {"status": "ok", "service": "SELF-RAG System – Self-Reflective Retrieval Augmented Generation"}
