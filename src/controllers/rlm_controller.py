"""Recursive Language Model controller with a bounded document REPL."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.environment.hybrid_retriever import HybridRetriever
from src.gemini_client import generate_json
from src.tools.read_tool import ReadTool
from src.tools.search_tool import SearchTool
from src.trace.tracer import TraceNode

logger = logging.getLogger(__name__)

ACTION_SYSTEM = """You are a Recursive Language Model operating inside a restricted Python REPL over an external document environment.

The full document lives outside the model. You should inspect it through tools, reason over small snippets, and use llm_query(...) for recursive sub-calls on smaller contexts.

You must return JSON with:
- "thought": a short note about the next step
- "code": Python code to execute in the REPL

Rules:
- Use only the provided tools and simple Python.
- Never import modules.
- Prefer small, inspectable steps.
- After search(...), usually inspect the best 1-3 hits with read_chunks(...) or read_span(...); avoid repeated blind re-searching.
- For multi-hop questions, solve each hop explicitly. If the question asks for a location, person, date, or number, do not stop at an intermediate entity.
- Only say the document lacks the answer after you have actually inspected plausible evidence.
- Use print(...) to expose the most important observation for the next step.
- When you have enough evidence, call finish(answer, confidence, reasoning).
- The final answer must be a short phrase, not a sentence.
- Confidence above 0.8 requires direct support from the inspected evidence.
- Do not fabricate facts that were not observed through tools or sub-calls."""

ACTION_PROMPT = """Question: {question}

Depth: {depth}/{max_depth}
Step: {step}/{max_steps}
Document stats: {doc_length} chars, {num_chunks} chunks

Available tools:
- search(query, top_k=5) -> list[{{chunk_id, start_char, end_char, preview}}]
- read_chunks(chunk_ids_or_results) -> text
- read_span(start_char, end_char) -> text
- chunk_span(chunk_id) -> (start_char, end_char)
- llm_query(question, context=None, chunk_ids=None, start=None, end=None) -> {{answer, confidence, reasoning}}
- finish(answer, confidence=0.0, reasoning="") -> marks the final answer

Persistent variables from earlier steps:
{locals_summary}

Recent execution history:
{history}

Last observation:
{last_observation}

Write the next Python code to execute."""

SYNTHESIS_SYSTEM = """You are a careful QA system. Given a question and an execution trace from a document REPL, produce the best supported final answer.

Return JSON with keys: answer, confidence, reasoning.
The answer field must be short and answer-only."""

SYNTHESIS_PROMPT = """Question: {question}

Execution trace:
{history}

Last observation:
{last_observation}

Produce the best supported final answer as JSON with keys answer, confidence, reasoning."""

BOOTSTRAP_QA_SYSTEM = """Answer the question using only the provided snippet.

Return JSON with keys: answer, confidence, reasoning.
- The answer must be a short phrase, not a sentence.
- If the snippet is insufficient, say so briefly and keep confidence below 0.3."""

BOOTSTRAP_QA_PROMPT = """Question: {question}

Snippet:
{evidence}

Answer using only this snippet."""

FINAL_VERIFY_SYSTEM = """You are verifying the final answer for a long-document QA system.

Return JSON with keys: answer, confidence, reasoning.

Rules:
- Answer the ORIGINAL question directly using only the provided evidence.
- Prefer an exact supported answer over vague summaries like "conflicting information".
- Do not switch the task into "most common", "most reliable", or "summarize the options" unless the original question explicitly asks for that.
- Only return an ambiguous answer if the evidence truly does not support a single direct answer.
- The answer must be a short phrase, not a sentence."""

FINAL_VERIFY_PROMPT = """Original question: {question}

Candidate answer: {candidate_answer}
Candidate confidence: {candidate_confidence:.2f}
Candidate reasoning: {candidate_reasoning}

Evidence gathered:
{evidence}

Execution trace summary:
{history}

Last observation:
{last_observation}

Return the best final answer to the original question as JSON with keys answer, confidence, reasoning."""


@dataclass
class AgentOutcome:
    answer: str
    confidence: float
    reasoning: str


class RLMController(BaseController):
    """Paper-style bounded REPL controller over an external document store."""

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
        self.max_steps = settings.rlm_max_steps
        self.history_chars = settings.rlm_history_chars
        self.max_chunks_per_step = settings.rlm_max_chunks_per_step
        self.initial_chunks = settings.rlm_initial_chunks

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

        trace = TraceNode(action="rlm", input=question, metadata={"mode": "repl"})
        result = self._repl_answer(
            question=question,
            store=store,
            search=search,
            read=read,
            depth=0,
            trace=trace,
            model=self.model,
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
                "controller_style": "repl_recursive",
            },
        )

    def _repl_answer(
        self,
        question: str,
        store: DocumentStore,
        search: SearchTool,
        read: ReadTool,
        depth: int,
        trace: TraceNode,
        model: str,
    ) -> AgentOutcome:
        """Run a bounded REPL over the document store."""
        history: list[dict[str, str]] = []
        namespace: dict[str, Any] = {"notes": ""}
        state: dict[str, Any] = {
            "final": None,
            "last_read": "",
            "history": history,
            "evidence_segments": [],
        }
        last_observation = self._bootstrap_workspace(
            question=question,
            store=store,
            search=search,
            read=read,
            trace=trace,
            depth=depth,
            namespace=namespace,
            state=state,
        )

        for step in range(1, self.max_steps + 1):
            decision = self._next_action(
                question=question,
                store=store,
                depth=depth,
                step=step,
                history=history,
                namespace=namespace,
                last_observation=last_observation,
                model=model,
            )
            thought = decision.get("thought", "").strip()
            code = self._strip_code_fence(decision.get("code", ""))

            plan_node = trace.add_child(TraceNode(
                action="plan",
                input=question,
                output=thought,
                metadata={"depth": depth, "step": step, "model": model},
            ))
            exec_node = plan_node.add_child(TraceNode(
                action="repl_exec",
                input=code,
                metadata={"depth": depth, "step": step},
            ))

            if not code:
                last_observation = "Planner returned empty code."
                exec_node.output = last_observation
                history.append({"thought": thought, "code": "", "observation": last_observation})
                continue

            last_observation = self._execute_code(
                code=code,
                question=question,
                store=store,
                search=search,
                read=read,
                trace=exec_node,
                depth=depth,
                namespace=namespace,
                state=state,
            )
            exec_node.output = last_observation
            history.append({"thought": thought, "code": code, "observation": last_observation})

            if state["final"] is not None:
                trace.metadata["steps"] = step
                return self._finalize_outcome(
                    question=question,
                    candidate=state["final"],
                    history=history,
                    last_observation=last_observation,
                    namespace=namespace,
                    state=state,
                    trace=trace,
                    depth=depth,
                    model=model,
                )

        trace.add_child(TraceNode(
            action="synthesize",
            input=question,
            metadata={"depth": depth, "steps": self.max_steps},
        ))
        final = self._synthesize_answer(question, history, last_observation, model)
        trace.metadata["steps"] = self.max_steps
        return self._finalize_outcome(
            question=question,
            candidate=final,
            history=history,
            last_observation=last_observation,
            namespace=namespace,
            state=state,
            trace=trace,
            depth=depth,
            model=model,
        )

    def _bootstrap_workspace(
        self,
        *,
        question: str,
        store: DocumentStore,
        search: SearchTool,
        read: ReadTool,
        trace: TraceNode,
        depth: int,
        namespace: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        """Seed the REPL with an initial retrieval pass over the full question."""
        bootstrap_k = max(1, self.initial_chunks)
        hits = search(question, top_k=bootstrap_k)
        evidence = read(hits)
        state["last_read"] = evidence
        self._remember_evidence(state, evidence)
        namespace["bootstrap_hits"] = [
            {
                "chunk_id": cid,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "preview": self._query_preview(chunk.text, question),
            }
            for cid in hits
            if (chunk := store.get_chunk(cid)) is not None
        ]
        namespace["bootstrap_evidence"] = evidence
        namespace["bootstrap_answer"] = self._snippet_answer(question, evidence)

        bootstrap_node = trace.add_child(TraceNode(
            action="bootstrap",
            input=question,
            metadata={"depth": depth, "top_k": bootstrap_k},
        ))
        bootstrap_node.add_child(TraceNode(
            action="search",
            input=question,
            output=str(hits),
            metadata={"depth": depth, "top_k": bootstrap_k},
        ))
        bootstrap_node.add_child(TraceNode(
            action="read",
            input=str(hits),
            output=f"({len(evidence)} chars)",
            metadata={"depth": depth, "num_chunks": len(hits)},
        ))
        return (
            "Bootstrap evidence is available in bootstrap_hits, bootstrap_evidence, and bootstrap_answer. "
            "Start from that before issuing many new searches."
        )

    def _next_action(
        self,
        *,
        question: str,
        store: DocumentStore,
        depth: int,
        step: int,
        history: list[dict[str, str]],
        namespace: dict[str, Any],
        last_observation: str,
        model: str,
    ) -> dict[str, str]:
        prompt = ACTION_PROMPT.format(
            question=question,
            depth=depth,
            max_depth=self.max_depth,
            step=step,
            max_steps=self.max_steps,
            doc_length=store.doc_length,
            num_chunks=store.num_chunks,
            locals_summary=self._summarize_namespace(namespace),
            history=self._format_history(history),
            last_observation=last_observation[-1200:],
        )
        try:
            result = self._coerce_mapping(
                generate_json(prompt, model=model, system=ACTION_SYSTEM, max_tokens=1200)
            )
        except Exception as exc:
            logger.warning("RLM action generation failed at depth=%d step=%d: %s", depth, step, exc)
            return {
                "thought": "Fallback action after planner formatting failure.",
                "code": (
                    f"hits = search({question!r}, top_k=3)\n"
                    "text = read_chunks(hits[:3])\n"
                    "print(text)"
                ),
            }
        return {
            "thought": str(result.get("thought", "")),
            "code": str(result.get("code", "")),
        }

    def _execute_code(
        self,
        *,
        code: str,
        question: str,
        store: DocumentStore,
        search: SearchTool,
        read: ReadTool,
        trace: TraceNode,
        depth: int,
        namespace: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        tools = self._build_tools(
            question=question,
            store=store,
            search=search,
            read=read,
            trace=trace,
            depth=depth,
            state=state,
        )
        exec_locals = dict(namespace)
        exec_locals.update(tools)

        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(code, {"__builtins__": self._safe_builtins()}, exec_locals)
        except Exception as exc:
            message = f"Execution error: {type(exc).__name__}: {exc}"
            trace.metadata["error"] = message
            return message

        namespace.clear()
        for key, value in exec_locals.items():
            if key in tools or key.startswith("__"):
                continue
            namespace[key] = value

        printed = stdout.getvalue().strip()
        summary = self._summarize_namespace(namespace)
        child_actions = [child.action for child in trace.children]
        if child_actions == ["search"]:
            suggestion = "Next step: inspect one of the returned hits with read_chunks(...) or read_span(...)."
        else:
            suggestion = ""
        if state["final"] is not None:
            final = state["final"]
            return (
                f"finish(answer={final.answer!r}, confidence={final.confidence:.2f}, "
                f"reasoning={final.reasoning!r})"
            )
        if printed and summary != "(no persistent variables)":
            response = f"Printed output:\n{printed}\n\nVariables:\n{summary}"
            if suggestion:
                response += f"\n\n{suggestion}"
            return response
        if printed:
            response = f"Printed output:\n{printed}"
            if suggestion:
                response += f"\n\n{suggestion}"
            return response
        response = f"Variables:\n{summary}"
        if suggestion:
            response += f"\n\n{suggestion}"
        return response

    def _build_tools(
        self,
        *,
        question: str,
        store: DocumentStore,
        search: SearchTool,
        read: ReadTool,
        trace: TraceNode,
        depth: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        def search_tool(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
            k = max(1, min(int(top_k or self.max_chunks_per_step), self.max_chunks_per_step))
            chunk_ids = search(query, top_k=k)
            results: list[dict[str, Any]] = []
            for cid in chunk_ids:
                chunk = store.get_chunk(cid)
                if chunk is None:
                    continue
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "preview": self._query_preview(chunk.text, query),
                })
            trace.add_child(TraceNode(
                action="search",
                input=query,
                output=str([r["chunk_id"] for r in results]),
                metadata={"depth": depth, "top_k": k},
            ))
            return results

        def read_chunks_tool(chunk_ids_or_results: Any) -> str:
            chunk_ids = self._normalize_chunk_ids(chunk_ids_or_results)
            text = read(chunk_ids)
            state["last_read"] = text
            self._remember_evidence(state, text)
            trace.add_child(TraceNode(
                action="read",
                input=str(chunk_ids),
                output=f"({len(text)} chars)",
                metadata={"depth": depth, "num_chunks": len(chunk_ids)},
            ))
            return text

        def read_span_tool(start_char: int, end_char: int) -> str:
            start = max(0, int(start_char))
            end = min(store.doc_length, int(end_char))
            if end < start:
                start, end = end, start
            text = read.read_span(start, end)
            state["last_read"] = text
            self._remember_evidence(state, text)
            trace.add_child(TraceNode(
                action="read_span",
                input=f"{start}:{end}",
                output=f"({len(text)} chars)",
                metadata={"depth": depth},
            ))
            return text

        def chunk_span_tool(chunk_id: int) -> tuple[int, int] | None:
            chunk = store.get_chunk(int(chunk_id))
            if chunk is None:
                return None
            return chunk.span

        def llm_query_tool(
            sub_question: str | None = None,
            *,
            question: str | None = None,
            context: str | None = None,
            chunk_ids: Any = None,
            start: int | None = None,
            end: int | None = None,
        ) -> dict[str, Any]:
            resolved_question = (sub_question or question or "").strip()
            if not resolved_question:
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "reasoning": "No sub-question provided to llm_query.",
                }
            if depth >= self.max_depth:
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "reasoning": "Maximum recursion depth reached.",
                }

            sub_context = self._resolve_sub_context(
                store=store,
                read=read,
                state=state,
                context=context,
                chunk_ids=chunk_ids,
                start=start,
                end=end,
            )
            if not sub_context.strip():
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "reasoning": "No context provided to llm_query.",
                }

            sub_trace = trace.add_child(TraceNode(
                action="llm_query",
                input=resolved_question,
                metadata={"depth": depth + 1, "context_chars": len(sub_context)},
            ))
            sub_store = DocumentStore(
                chunk_size=store.chunk_size,
                chunk_overlap=store.chunk_overlap,
                min_chunk_size=store.min_chunk_size,
            )
            sub_store.ingest(sub_context, doc_id=f"{store.doc_id}_sub_{depth + 1}")
            sub_retriever = HybridRetriever(sub_store, cache_key="")
            sub_search = SearchTool(sub_retriever)
            sub_read = ReadTool(sub_store)
            sub_result = self._repl_answer(
                question=resolved_question,
                store=sub_store,
                search=sub_search,
                read=sub_read,
                depth=depth + 1,
                trace=sub_trace,
                model=self.model,
            )
            sub_trace.output = sub_result.answer
            sub_trace.metadata["confidence"] = sub_result.confidence
            return {
                "answer": sub_result.answer,
                "confidence": sub_result.confidence,
                "reasoning": sub_result.reasoning,
            }

        def finish_tool(answer: str, confidence: float = 0.0, reasoning: str = "") -> None:
            clean_answer = self._normalize_answer(str(answer).strip())
            clean_confidence = max(0.0, min(1.0, float(confidence)))
            if "document does not contain" in clean_answer.lower() or "cannot determine" in clean_answer.lower():
                clean_confidence = min(clean_confidence, 0.1)
            final = AgentOutcome(
                answer=clean_answer,
                confidence=clean_confidence,
                reasoning=str(reasoning).strip(),
            )
            state["final"] = final
            trace.add_child(TraceNode(
                action="finish",
                input=question,
                output=final.answer,
                metadata={"depth": depth, "confidence": final.confidence},
            ))

        return {
            "search": search_tool,
            "read_chunks": read_chunks_tool,
            "read_span": read_span_tool,
            "chunk_span": chunk_span_tool,
            "llm_query": llm_query_tool,
            "finish": finish_tool,
        }

    def _resolve_sub_context(
        self,
        *,
        store: DocumentStore,
        read: ReadTool,
        state: dict[str, Any],
        context: str | None,
        chunk_ids: Any,
        start: int | None,
        end: int | None,
    ) -> str:
        if context is not None:
            return str(context)
        if chunk_ids is not None:
            return read(self._normalize_chunk_ids(chunk_ids))
        if start is not None and end is not None:
            return read.read_span(int(start), int(end))
        return str(state.get("last_read", ""))

    def _finalize_outcome(
        self,
        *,
        question: str,
        candidate: AgentOutcome,
        history: list[dict[str, str]],
        last_observation: str,
        namespace: dict[str, Any],
        state: dict[str, Any],
        trace: TraceNode,
        depth: int,
        model: str,
    ) -> AgentOutcome:
        if depth > 0:
            return candidate

        evidence = self._build_verification_evidence(namespace, state)
        if not evidence:
            return candidate

        verify_node = trace.add_child(TraceNode(
            action="verify_final",
            input=question,
            metadata={"depth": depth, "candidate_confidence": candidate.confidence},
        ))
        try:
            verified = self._verify_final_answer(
                question=question,
                candidate=candidate,
                evidence=evidence,
                history=history,
                last_observation=last_observation,
                model=model,
            )
        except Exception as exc:
            logger.warning("RLM final verification failed at depth=%d: %s", depth, exc)
            verify_node.output = f"verification_failed: {exc}"
            verify_node.metadata["error"] = str(exc)
            return candidate

        verify_node.output = verified.answer
        verify_node.metadata["confidence"] = verified.confidence

        if not verified.answer:
            return candidate
        if self._is_ambiguous_answer(verified.answer) and not self._is_ambiguous_answer(candidate.answer):
            return candidate
        return verified

    def _verify_final_answer(
        self,
        *,
        question: str,
        candidate: AgentOutcome,
        evidence: str,
        history: list[dict[str, str]],
        last_observation: str,
        model: str,
    ) -> AgentOutcome:
        prompt = FINAL_VERIFY_PROMPT.format(
            question=question,
            candidate_answer=candidate.answer or "(empty)",
            candidate_confidence=candidate.confidence,
            candidate_reasoning=candidate.reasoning or "(none)",
            evidence=evidence,
            history=self._format_history(history),
            last_observation=last_observation[-1200:] or "(none)",
        )
        result = self._coerce_mapping(
            generate_json(prompt, model=model, system=FINAL_VERIFY_SYSTEM, max_tokens=400)
        )
        return AgentOutcome(
            answer=self._normalize_answer(str(result.get("answer", "")).strip()),
            confidence=self._coerce_confidence(result.get("confidence", candidate.confidence)),
            reasoning=str(result.get("reasoning", "")).strip(),
        )

    def _synthesize_answer(
        self,
        question: str,
        history: list[dict[str, str]],
        last_observation: str,
        model: str,
    ) -> AgentOutcome:
        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            history=self._format_history(history),
            last_observation=last_observation[-1200:],
        )
        result = self._coerce_mapping(
            generate_json(prompt, model=model, system=SYNTHESIS_SYSTEM, max_tokens=600)
        )
        return AgentOutcome(
            answer=str(result.get("answer", "")).strip(),
            confidence=self._coerce_confidence(result.get("confidence", 0.0)),
            reasoning=str(result.get("reasoning", "")).strip(),
        )

    def _snippet_answer(self, question: str, evidence: str) -> dict[str, Any]:
        prompt = BOOTSTRAP_QA_PROMPT.format(question=question, evidence=evidence[:12000])
        result = self._coerce_mapping(
            generate_json(prompt, model=self.model, system=BOOTSTRAP_QA_SYSTEM, max_tokens=300)
        )
        return {
            "answer": self._normalize_answer(str(result.get("answer", "")).strip()),
            "confidence": self._coerce_confidence(result.get("confidence", 0.0)),
            "reasoning": str(result.get("reasoning", "")).strip(),
        }

    def _format_history(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "(no prior steps)"
        blocks = []
        for idx, entry in enumerate(history[-4:], start=max(1, len(history) - 3)):
            blocks.append(
                f"Step {idx}\n"
                f"Thought: {entry['thought']}\n"
                f"Code:\n{self._clip(entry['code'], 500)}\n"
                f"Observation:\n{self._clip(entry['observation'], 900)}"
            )
        text = "\n\n".join(blocks)
        return text[-self.history_chars:]

    def _summarize_namespace(self, namespace: dict[str, Any]) -> str:
        visible = []
        for key in sorted(namespace):
            if key.startswith("_"):
                continue
            value = namespace[key]
            if callable(value):
                continue
            visible.append(f"{key} = {self._summarize_value(value)}")
        if not visible:
            return "(no persistent variables)"
        return "\n".join(visible)[-2000:]

    def _summarize_value(self, value: Any) -> str:
        if isinstance(value, str):
            return repr(self._clip(value, 220))
        if isinstance(value, list):
            preview = ", ".join(self._summarize_value(v) for v in value[:3])
            more = "" if len(value) <= 3 else f", ... ({len(value)} items)"
            return f"[{preview}{more}]"
        if isinstance(value, dict):
            items = list(value.items())[:4]
            preview = ", ".join(f"{k!r}: {self._summarize_value(v)}" for k, v in items)
            more = "" if len(value) <= 4 else ", ..."
            return "{" + preview + more + "}"
        return self._clip(repr(value), 220)

    def _remember_evidence(self, state: dict[str, Any], text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        segments = [segment.strip() for segment in cleaned.split("\n\n---\n\n") if segment.strip()]
        if not segments:
            segments = [cleaned]
        evidence_segments = state.setdefault("evidence_segments", [])
        for segment in segments:
            clipped = self._clip(segment, 2500)
            if clipped not in evidence_segments:
                evidence_segments.append(clipped)

    def _build_verification_evidence(
        self,
        namespace: dict[str, Any],
        state: dict[str, Any],
        max_chars: int = 18000,
    ) -> str:
        segments: list[str] = []
        bootstrap_evidence = str(namespace.get("bootstrap_evidence", "")).strip()
        if bootstrap_evidence:
            segments.extend(
                segment.strip()
                for segment in bootstrap_evidence.split("\n\n---\n\n")
                if segment.strip()
            )
        segments.extend(str(segment).strip() for segment in state.get("evidence_segments", []) if str(segment).strip())

        unique_segments: list[str] = []
        for segment in segments:
            clipped = self._clip(segment, 2500)
            if clipped and clipped not in unique_segments:
                unique_segments.append(clipped)

        assembled: list[str] = []
        total = 0
        for segment in unique_segments:
            addition = len(segment) + (7 if assembled else 0)
            if total + addition > max_chars:
                break
            assembled.append(segment)
            total += addition
        return "\n\n---\n\n".join(assembled)

    def _normalize_chunk_ids(self, value: Any) -> list[int]:
        if isinstance(value, int):
            return [value]
        if isinstance(value, dict):
            cid = value.get("chunk_id")
            return [int(cid)] if cid is not None else []
        if isinstance(value, (list, tuple)):
            chunk_ids: list[int] = []
            for item in value:
                chunk_ids.extend(self._normalize_chunk_ids(item))
            return chunk_ids
        return []

    def _strip_code_fence(self, code: str) -> str:
        cleaned = code.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _clip(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _query_preview(self, text: str, query: str) -> str:
        text_lower = text.lower()
        for token in query.lower().split():
            if len(token) < 4:
                continue
            idx = text_lower.find(token)
            if idx >= 0:
                start = max(0, idx - 80)
                end = min(len(text), idx + 120)
                return self._clip(text[start:end], 220)
        return self._clip(text, 180)

    def _normalize_answer(self, answer: str) -> str:
        trimmed = answer.strip().rstrip(".")
        lowered = trimmed.lower()
        patterns = [
            " should be ",
            " is ",
            " was ",
            " were ",
            " are ",
            " equals ",
            " amounted to ",
        ]
        for pattern in patterns:
            idx = lowered.find(pattern)
            if idx > 0:
                candidate = trimmed[idx + len(pattern):].strip(" .")
                if candidate:
                    return candidate
        return trimmed

    def _is_ambiguous_answer(self, answer: str) -> bool:
        lowered = answer.strip().lower()
        ambiguous_markers = (
            "conflicting",
            "multiple",
            "ambiguous",
            "cannot determine",
            "can't determine",
            "unclear",
            "no definitive",
            "not enough information",
        )
        return any(marker in lowered for marker in ambiguous_markers)

    def _coerce_confidence(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        text = str(value).strip().lower()
        label_map = {"high": 0.85, "medium": 0.5, "low": 0.2}
        if text in label_map:
            return label_map[text]
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return 0.0

    def _coerce_mapping(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value[0]
        return {}

    def _safe_builtins(self) -> dict[str, Any]:
        return {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
