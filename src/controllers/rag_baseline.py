"""Single-pass RAG baseline: retrieve → answer."""

from __future__ import annotations

from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.environment.hybrid_retriever import HybridRetriever
from src.tools.read_tool import ReadTool
from src.tools.reason_tool import ReasonTool
from src.tools.search_tool import SearchTool
from src.trace.tracer import TraceNode


class RAGController(BaseController):
    """Single-pass retrieve-and-generate."""

    method_name = "rag"

    def __init__(self, model: str | None = None, top_k: int | None = None):
        self.model = model or settings.model_fast
        self.top_k = top_k or settings.rag_top_k

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        trace = TraceNode(action="rag", input=question)

        retriever = kwargs.get("retriever")
        if retriever is None:
            retriever = HybridRetriever(store, cache_key=kwargs.get("cache_key", ""))

        search = SearchTool(retriever)
        read = ReadTool(store)
        reason = ReasonTool(model=self.model)

        # Search
        chunk_ids = search(question, top_k=self.top_k)
        trace.add_child(TraceNode(
            action="search", input=question,
            output=str(chunk_ids),
        ))

        # Read
        evidence = read(chunk_ids)
        trace.add_child(TraceNode(
            action="read", input=str(chunk_ids),
            output=f"({len(evidence)} chars)",
        ))

        # Reason
        result = reason.reason(question, evidence)
        trace.add_child(TraceNode(
            action="reason", input=question,
            output=result.answer,
            metadata={"confidence": result.confidence},
        ))

        trace.output = result.answer
        trace.metadata["confidence"] = result.confidence

        return ControllerResult(
            answer=result.answer,
            confidence=result.confidence,
            method=self.method_name,
            trace=trace,
            metadata={"reasoning": result.reasoning},
        )
