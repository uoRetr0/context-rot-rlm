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
            store=store,
            depth=0,
            trace=trace,
            accumulated_chunk_ids=set(),
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

    def _get_top_k(self, depth: int) -> int:
        """Return chunk count based on recursion depth."""
        if depth == 0:
            return settings.rlm_initial_chunks
        return settings.rlm_sub_question_chunks

    def _expand_neighbors(
        self, chunk_ids: list[int], store: DocumentStore, top_n: int = 3
    ) -> list[int]:
        """Expand the top-N chunks by including their left/right neighbors."""
        expanded = set(chunk_ids)
        num_chunks = store.num_chunks
        for cid in chunk_ids[:top_n]:
            if cid - 1 >= 0:
                expanded.add(cid - 1)
            if cid + 1 < num_chunks:
                expanded.add(cid + 1)
        # Preserve original ordering, append new neighbors at end
        ordered = list(chunk_ids)
        for cid in sorted(expanded - set(chunk_ids)):
            ordered.append(cid)
        return ordered

    def _deduplicate_chunks(
        self,
        new_chunk_ids: list[int],
        accumulated: set[int],
        top_k: int,
    ) -> list[int]:
        """Prioritize unseen chunks, backfill with seen ones if needed."""
        unseen = [cid for cid in new_chunk_ids if cid not in accumulated]
        seen = [cid for cid in new_chunk_ids if cid in accumulated]
        result = unseen[:top_k]
        if len(result) < top_k:
            result.extend(seen[: top_k - len(result)])
        return result

    def _recursive_answer(
        self,
        question: str,
        search: SearchTool,
        read: ReadTool,
        reason: ReasonTool,
        store: DocumentStore,
        depth: int,
        trace: TraceNode,
        accumulated_chunk_ids: set[int],
        seed_chunk_ids: list[int] | None = None,
    ) -> ReasonResult:
        """Core recursive algorithm."""
        top_k = self._get_top_k(depth)

        # Step 1: SEARCH
        raw_chunk_ids = search(question, top_k=top_k + 5)  # fetch extra for dedup
        if seed_chunk_ids:
            raw_chunk_ids = list(dict.fromkeys(seed_chunk_ids + raw_chunk_ids))
        chunk_ids = self._deduplicate_chunks(raw_chunk_ids, accumulated_chunk_ids, top_k)
        new_chunks = len([cid for cid in chunk_ids if cid not in accumulated_chunk_ids])
        accumulated_chunk_ids.update(chunk_ids)

        trace.add_child(TraceNode(
            action="search", input=question,
            output=str(chunk_ids),
            metadata={
                "depth": depth,
                "top_k": top_k,
                "new_chunks": new_chunks,
                "seeded_chunks": len(seed_chunk_ids or []),
            },
        ))

        # Step 1b: Context expansion at depth 0 — read neighbors of top chunks
        if depth == 0:
            chunk_ids = self._expand_neighbors(chunk_ids, store, top_n=3)
            accumulated_chunk_ids.update(chunk_ids)

        # Step 2: READ
        evidence = read(chunk_ids)
        trace.add_child(TraceNode(
            action="read", input=str(chunk_ids),
            output=f"({len(evidence)} chars)",
            metadata={"depth": depth, "num_chunks": len(chunk_ids)},
        ))

        # Step 3: REASON
        result = reason.reason(question, evidence)
        reason_node = TraceNode(
            action="reason", input=question,
            output=result.answer,
            metadata={"confidence": result.confidence, "depth": depth},
        )
        trace.add_child(reason_node)

        # Step 3b: Refinement re-search at depth 0 for medium confidence
        if depth == 0 and 0.3 <= result.confidence <= 0.5:
            refined_query = f"{question} {result.answer}"
            refined_chunk_ids_raw = search(refined_query, top_k=top_k)
            refined_chunk_ids = self._deduplicate_chunks(
                refined_chunk_ids_raw, accumulated_chunk_ids, top_k
            )
            if refined_chunk_ids:
                accumulated_chunk_ids.update(refined_chunk_ids)
                combined_ids = list(dict.fromkeys(chunk_ids + refined_chunk_ids))
                evidence = read(combined_ids)
                result = reason.reason(question, evidence)
                trace.add_child(TraceNode(
                    action="refine_search", input=refined_query,
                    output=str(refined_chunk_ids),
                    metadata={"depth": depth, "new_chunks": len(refined_chunk_ids)},
                ))
                trace.add_child(TraceNode(
                    action="refine_reason", input=question,
                    output=result.answer,
                    metadata={"confidence": result.confidence, "depth": depth},
                ))

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

        # Decompose into sub-questions with evidence context
        logger.info(
            "RLM depth=%d: confidence=%.2f < %.2f, decomposing...",
            depth, result.confidence, self.confidence_threshold,
        )
        # Build evidence summary for decompose
        evidence_summary = f"Current answer: {result.answer}\nReasoning: {result.reasoning}\nEvidence chunks used: {result.evidence_used}"

        sub_questions, search_queries = reason.decompose(
            question, result.answer, result.confidence,
            evidence_summary=evidence_summary,
        )
        sub_questions = sub_questions[: self.max_sub_questions]
        search_queries = search_queries[: self.max_sub_questions]

        decompose_node = TraceNode(
            action="decompose", input=question,
            output=str(sub_questions),
            metadata={
                "depth": depth,
                "num_sub_questions": len(sub_questions),
                "search_queries": search_queries,
            },
        )
        trace.add_child(decompose_node)

        # Recurse on each sub-question
        sub_answers = []
        for i, sq in enumerate(sub_questions):
            sub_trace = TraceNode(action="sub_rlm", input=sq, metadata={"depth": depth + 1})
            trace.add_child(sub_trace)

            # Use optimized search query if available — pre-search to seed chunks
            seed_ids: list[int] = []
            if i < len(search_queries) and search_queries[i] != sq:
                seed_ids = search(search_queries[i], top_k=self._get_top_k(depth + 1))
                sub_trace.add_child(TraceNode(
                    action="seed_search",
                    input=search_queries[i],
                    output=str(seed_ids),
                    metadata={"depth": depth + 1},
                ))

            sub_result = self._recursive_answer(
                question=sq,
                search=search,
                read=read,
                reason=reason,
                store=store,
                depth=depth + 1,
                trace=sub_trace,
                accumulated_chunk_ids=accumulated_chunk_ids,
                seed_chunk_ids=seed_ids,
            )

            sub_trace.output = sub_result.answer
            sub_trace.metadata["confidence"] = sub_result.confidence

            sub_answers.append({
                "question": sq,
                "answer": sub_result.answer,
                "confidence": sub_result.confidence,
                "reasoning": sub_result.reasoning,
                "evidence_used": sub_result.evidence_used,
            })

        # Step 5: MERGE with full context
        merged = reason.merge(
            question,
            result.answer,
            sub_answers,
            initial_confidence=result.confidence,
            initial_reasoning=result.reasoning,
        )
        merge_node = TraceNode(
            action="merge", input=f"{len(sub_answers)} sub-answers",
            output=merged.answer,
            metadata={"confidence": merged.confidence, "depth": depth},
        )
        trace.add_child(merge_node)

        return merged
