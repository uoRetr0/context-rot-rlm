"""Recursive Language Model (RLM) controller — the core algorithm."""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.cost_tracker import tracker
from src.environment.document_store import DocumentStore
from src.environment.hybrid_retriever import HybridRetriever
from src.tools.read_tool import ReadTool
from src.tools.reason_tool import ReasonResult, ReasonTool
from src.tools.search_tool import SearchTool
from src.trace.tracer import TraceNode

logger = logging.getLogger(__name__)


class RLMController(BaseController):
    """Recursive retrieval controller that decomposes questions on low confidence."""

    method_name = "rlm"

    def __init__(
        self,
        model: str | None = None,
        max_depth: int | None = None,
        confidence_threshold: float | None = None,
        max_sub_questions: int | None = None,
    ):
        self.model = model or settings.model_fast
        self.max_depth = max_depth if max_depth is not None else settings.rlm_max_depth
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.rlm_confidence_threshold
        )
        self.max_sub_questions = (
            max_sub_questions
            if max_sub_questions is not None
            else settings.rlm_max_sub_questions
        )

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        retriever = kwargs.get("retriever")
        if retriever is None:
            retriever = HybridRetriever(store, cache_key=kwargs.get("cache_key", ""))

        search = SearchTool(retriever)
        read = ReadTool(store)
        reason = ReasonTool(model=self.model)

        trace = TraceNode(action="rlm", input=question)
        result = self._recursive_answer(
            question=question,
            search=search,
            read=read,
            reason=reason,
            depth=0,
            trace=trace,
        )

        trace.output = result.answer
        trace.metadata["confidence"] = result.confidence
        trace.metadata["final_depth"] = trace.max_depth()

        return ControllerResult(
            answer=result.answer,
            confidence=result.confidence,
            method=self.method_name,
            trace=trace,
            metadata={
                "reasoning": result.reasoning,
                "max_depth_reached": trace.max_depth(),
            },
        )

    def _recursive_answer(
        self,
        question: str,
        search: SearchTool,
        read: ReadTool,
        reason: ReasonTool,
        depth: int,
        trace: TraceNode,
    ) -> ReasonResult:
        """Core recursive algorithm."""
        # Step 1: SEARCH
        chunk_ids = search(question, top_k=settings.rlm_max_chunks_per_step)
        trace.add_child(TraceNode(
            action="search", input=question,
            output=str(chunk_ids),
            metadata={"depth": depth},
        ))

        # Step 2: READ
        evidence = read(chunk_ids)
        trace.add_child(TraceNode(
            action="read", input=str(chunk_ids),
            output=f"({len(evidence)} chars)",
            metadata={"depth": depth},
        ))

        # Step 3: REASON
        result = reason.reason(question, evidence)
        reason_node = TraceNode(
            action="reason", input=question,
            output=result.answer,
            metadata={"confidence": result.confidence, "depth": depth},
        )
        trace.add_child(reason_node)

        # Step 4: Check confidence and recurse
        should_recurse = (
            result.confidence < self.confidence_threshold
            and depth < self.max_depth
        )

        # Budget check
        try:
            tracker.check_budget()
        except Exception:
            should_recurse = False

        if not should_recurse:
            return result

        # Decompose into sub-questions
        logger.info(
            "RLM depth=%d: confidence=%.2f < %.2f, decomposing...",
            depth, result.confidence, self.confidence_threshold,
        )
        sub_questions = reason.decompose(question, result.answer, result.confidence)
        sub_questions = sub_questions[: self.max_sub_questions]

        decompose_node = TraceNode(
            action="decompose", input=question,
            output=str(sub_questions),
            metadata={"depth": depth, "num_sub_questions": len(sub_questions)},
        )
        trace.add_child(decompose_node)

        # Recurse on each sub-question
        sub_answers = []
        for sq in sub_questions:
            sub_trace = TraceNode(action="sub_rlm", input=sq, metadata={"depth": depth + 1})
            trace.add_child(sub_trace)

            sub_result = self._recursive_answer(
                question=sq,
                search=search,
                read=read,
                reason=reason,
                depth=depth + 1,
                trace=sub_trace,
            )
            sub_trace.output = sub_result.answer
            sub_trace.metadata["confidence"] = sub_result.confidence

            sub_answers.append({
                "question": sq,
                "answer": sub_result.answer,
                "confidence": sub_result.confidence,
            })

        # Step 5: MERGE
        merged = reason.merge(question, result.answer, sub_answers)
        merge_node = TraceNode(
            action="merge", input=f"{len(sub_answers)} sub-answers",
            output=merged.answer,
            metadata={"confidence": merged.confidence, "depth": depth},
        )
        trace.add_child(merge_node)

        return merged
