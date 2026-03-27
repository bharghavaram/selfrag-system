# SELF-RAG System

> Implements the SELF-RAG paper (Asai et al., 2023) — LLM that decides when to retrieve, critiques its own outputs, and self-corrects

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Paper](https://img.shields.io/badge/ArXiv-2310.11511-red)](https://arxiv.org/abs/2310.11511)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-orange)](https://faiss.ai)

## Overview

SELF-RAG is a cutting-edge RAG framework where the LLM **learns to critique itself** using special reflection tokens. Unlike standard RAG, the model actively decides whether retrieval is needed, evaluates retrieved passage quality, checks if its response is factually supported, and iteratively self-corrects.

## Reflection Tokens (SELF-RAG Paper)

| Token | Purpose | Scale |
|-------|---------|-------|
| `RETRIEVE` | Should the model retrieve? | yes/no |
| `IS_REL` | Is this passage relevant? | 0.0 – 1.0 |
| `IS_SUP` | Is the response supported by evidence? | 0.0 – 1.0 |
| `IS_USE` | How useful is this response? | 1 – 5 |

## Workflow

```
Question
   ↓
[RETRIEVE] Decide if retrieval is needed
   ↓ (if yes)
Retrieve top-K passages from FAISS
   ↓
[IS_REL] Filter irrelevant passages (threshold: 0.7)
   ↓
Generate response using relevant passages
   ↓
[IS_SUP] Check factual support → flag unsupported claims
   ↓
[IS_USE] Rate utility → identify improvements
   ↓
Self-Reflection Loop (max 3 rounds) → improve if needed
   ↓
Final verified response with full trace
```

## Quick Start

```bash
git clone https://github.com/bharghavaram/selfrag-system
cd selfrag-system
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/selfrag/ingest` | Upload documents to knowledge base |
| POST | `/api/v1/selfrag/query` | Query with full SELF-RAG pipeline |

### Example Response

```json
{
  "question": "What is SELF-RAG?",
  "answer": "SELF-RAG is...",
  "retrieved": true,
  "relevant_passages": 3,
  "scores": {"relevance": 0.85, "support": 0.92, "utility": 4},
  "reflection_rounds": 1,
  "unsupported_claims": [],
  "trace": [...]
}
```

## Reference

> Asai, A., Wu, Z., Wang, B., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ArXiv:2310.11511
