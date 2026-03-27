"""
SELF-RAG System – Implements the SELF-RAG paper (Asai et al., 2023).
LLM decides when to retrieve, critiques its own outputs using reflection tokens,
and selects the best response using IS_REL, IS_SUP, IS_USE scoring.
"""
import logging
import json
from pathlib import Path
from typing import Optional, List
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings

logger = logging.getLogger(__name__)

# SELF-RAG Special Tokens (simulated as prompts)
RETRIEVE_DECISION_PROMPT = """Should retrieval be used to answer this question?
Question: {question}
Answer with JSON: {{"should_retrieve": true/false, "reasoning": "..."}}
Consider: factual questions → retrieve, simple/conversational → no retrieve."""

RELEVANCE_PROMPT = """Rate the relevance of this retrieved passage to the question.
Question: {question}
Passage: {passage}
JSON: {{"is_relevant": true/false, "relevance_score": 0.0-1.0, "reasoning": "..."}}"""

GENERATION_PROMPT = """Answer the question using the provided context. Be factual and precise.
Question: {question}
Context: {context}
Generate a comprehensive answer:"""

SUPPORT_CHECK_PROMPT = """Does the generated response accurately reflect the supporting passages?
Question: {question}
Response: {response}
Supporting Passages:
{passages}
JSON: {{"is_supported": true/false, "support_score": 0.0-1.0, "unsupported_claims": [...], "reasoning": "..."}}"""

UTILITY_PROMPT = """Rate the overall utility of this response for answering the question.
Question: {question}
Response: {response}
JSON: {{"utility_score": 1-5, "is_useful": true/false, "improvements": [...], "reasoning": "..."}}"""

CRITIQUE_PROMPT = """You previously generated this response. Critique it and improve it.
Question: {question}
Original Response: {response}
Issues Found: {issues}
Generate an improved response that fixes these issues:"""


class SELFRAGService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.vectorstore: Optional[FAISS] = None
        self._load_index()

    def _load_index(self):
        index_path = Path(settings.FAISS_INDEX_PATH)
        if index_path.exists():
            self.vectorstore = FAISS.load_local(str(index_path), self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded")

    def _call_llm(self, prompt: str, json_mode: bool = False, temperature: float = None) -> str:
        kwargs = {
            "model": settings.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _decide_retrieve(self, question: str) -> dict:
        result = self._call_llm(RETRIEVE_DECISION_PROMPT.format(question=question), json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            return {"should_retrieve": True, "reasoning": "Default to retrieval"}

    def _retrieve(self, question: str, k: int = None) -> List[dict]:
        if not self.vectorstore:
            return []
        k = k or settings.MAX_RETRIEVE
        docs = self.vectorstore.similarity_search_with_score(question, k=k)
        return [{"content": doc.page_content, "score": float(score), "metadata": doc.metadata} for doc, score in docs]

    def _check_relevance(self, question: str, passage: str) -> dict:
        result = self._call_llm(RELEVANCE_PROMPT.format(question=question, passage=passage[:500]), json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            return {"is_relevant": True, "relevance_score": 0.5, "reasoning": ""}

    def _generate(self, question: str, context: str) -> str:
        return self._call_llm(GENERATION_PROMPT.format(question=question, context=context), temperature=0.3)

    def _check_support(self, question: str, response: str, passages: List[str]) -> dict:
        passages_text = "\n---\n".join(passages[:3])
        result = self._call_llm(SUPPORT_CHECK_PROMPT.format(
            question=question, response=response[:500], passages=passages_text[:1500]
        ), json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            return {"is_supported": True, "support_score": 0.5, "unsupported_claims": [], "reasoning": ""}

    def _check_utility(self, question: str, response: str) -> dict:
        result = self._call_llm(UTILITY_PROMPT.format(question=question, response=response[:500]), json_mode=True)
        try:
            return json.loads(result)
        except Exception:
            return {"utility_score": 3, "is_useful": True, "improvements": [], "reasoning": ""}

    def _critique_and_improve(self, question: str, response: str, issues: List[str]) -> str:
        return self._call_llm(CRITIQUE_PROMPT.format(
            question=question, response=response, issues="\n".join(issues)
        ), temperature=0.4)

    def query(self, question: str) -> dict:
        trace = []

        # Step 1: Retrieve decision (RETRIEVE token)
        retrieve_decision = self._decide_retrieve(question)
        trace.append({"step": "retrieve_decision", "result": retrieve_decision})

        if not retrieve_decision.get("should_retrieve", True):
            # No retrieval — direct generation
            response = self._generate(question, "No external context (direct knowledge)")
            utility = self._check_utility(question, response)
            return {
                "question": question,
                "answer": response,
                "retrieved": False,
                "scores": {"utility": utility.get("utility_score", 0), "support": None, "relevance": None},
                "reflection_rounds": 0,
                "trace": trace,
            }

        # Step 2: Retrieve passages
        passages = self._retrieve(question)
        trace.append({"step": "retrieval", "passages_found": len(passages)})

        # Step 3: Filter by relevance (IS_REL token)
        relevant_passages = []
        for p in passages:
            rel = self._check_relevance(question, p["content"])
            trace.append({"step": "relevance_check", "score": rel.get("relevance_score", 0)})
            if rel.get("is_relevant") and rel.get("relevance_score", 0) >= settings.RELEVANCE_THRESHOLD:
                relevant_passages.append(p)

        if not relevant_passages:
            relevant_passages = passages[:2]  # Fallback: use top 2

        context = "\n\n".join([p["content"] for p in relevant_passages])

        # Step 4: Generate response
        response = self._generate(question, context)
        trace.append({"step": "generation", "response_length": len(response)})

        # Step 5: Support check (IS_SUP token)
        passage_texts = [p["content"] for p in relevant_passages]
        support = self._check_support(question, response, passage_texts)
        trace.append({"step": "support_check", "score": support.get("support_score", 0)})

        # Step 6: Utility check (IS_USE token)
        utility = self._check_utility(question, response)
        trace.append({"step": "utility_check", "score": utility.get("utility_score", 0)})

        # Step 7: Self-reflection & improvement loop
        reflection_rounds = 0
        for round_num in range(settings.MAX_REFLECTION_ROUNDS):
            issues = []
            if not support.get("is_supported") or support.get("support_score", 1) < settings.SUPPORT_THRESHOLD:
                issues.extend(support.get("unsupported_claims", ["Response not supported by passages"]))
            if utility.get("utility_score", 5) < 3:
                issues.extend(utility.get("improvements", ["Response needs improvement"]))

            if not issues:
                break

            reflection_rounds += 1
            improved = self._critique_and_improve(question, response, issues)
            support = self._check_support(question, improved, passage_texts)
            utility = self._check_utility(question, improved)
            response = improved
            trace.append({"step": f"reflection_round_{round_num + 1}", "issues": issues, "new_support": support.get("support_score"), "new_utility": utility.get("utility_score")})

        return {
            "question": question,
            "answer": response,
            "retrieved": True,
            "relevant_passages": len(relevant_passages),
            "scores": {
                "relevance": sum(p.get("score", 0) for p in relevant_passages) / max(len(relevant_passages), 1),
                "support": support.get("support_score", 0),
                "utility": utility.get("utility_score", 0),
            },
            "reflection_rounds": reflection_rounds,
            "unsupported_claims": support.get("unsupported_claims", []),
            "trace": trace,
        }

    def ingest(self, file_paths: List[str]) -> dict:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_chunks = []
        for fp in file_paths:
            try:
                loader = PyPDFLoader(fp) if fp.endswith(".pdf") else TextLoader(fp)
                chunks = splitter.split_documents(loader.load())
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error("Ingest error for %s: %s", fp, exc)

        if all_chunks:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(all_chunks)
            index_path = Path(settings.FAISS_INDEX_PATH)
            index_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(index_path))

        return {"chunks_indexed": len(all_chunks), "total_in_index": self.vectorstore.index.ntotal if self.vectorstore else 0}


_service: Optional[SELFRAGService] = None
def get_selfrag_service() -> SELFRAGService:
    global _service
    if _service is None:
        _service = SELFRAGService()
    return _service
