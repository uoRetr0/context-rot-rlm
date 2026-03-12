"""Full-context baseline: send entire document to LLM."""

from __future__ import annotations

from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.gemini_client import generate_json
from src.trace.tracer import TraceNode

SYSTEM = """You are a precise question-answering system. Answer the question based ONLY on the provided document.

Respond with JSON: {"answer": "...", "confidence": 0.0-1.0, "reasoning": "..."}

IMPORTANT:
- The "answer" field must be SHORT: just the final fact, name, place, number, or brief phrase.
- Do not return a full sentence in the "answer" field.
- Put any explanation in "reasoning"."""

PROMPT = """Document:
{document}

Question: {question}

Answer based only on the document above.
IMPORTANT: The "answer" field must be SHORT and answer-only, not a full sentence.
Respond as JSON with keys: answer, confidence, reasoning."""


class FullContextController(BaseController):
    """Sends the entire document as context to the LLM."""

    method_name = "fullcontext"
    requires_retriever = False

    def __init__(self, model: str | None = None):
        self.model = model or settings.model_fast

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        trace = TraceNode(action="fullcontext", input=question)

        doc_text = store.full_text
        # Truncate if needed (by word count)
        words = doc_text.split()
        if len(words) > settings.fullcontext_max_tokens:
            doc_text = " ".join(words[: settings.fullcontext_max_tokens])
            trace.add_child(TraceNode(action="truncate", input=f"{len(words)} -> {settings.fullcontext_max_tokens} words"))

        prompt = PROMPT.format(document=doc_text, question=question)
        result = generate_json(prompt, model=self.model, system=SYSTEM)

        trace.output = result.get("answer", "")
        trace.metadata["confidence"] = result.get("confidence", 0.0)

        return ControllerResult(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            method=self.method_name,
            trace=trace,
            metadata={"reasoning": result.get("reasoning", "")},
        )
