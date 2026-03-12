"""Map-reduce baseline: map chunks → reduce answers."""

from __future__ import annotations

from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.environment.hybrid_retriever import HybridRetriever
from src.gemini_client import generate_json
from src.tools.search_tool import SearchTool
from src.trace.tracer import TraceNode

MAP_SYSTEM = """You answer questions using ONLY the provided text chunk.
Respond with JSON: {"answer": "...", "confidence": 0.0-1.0, "relevant": true/false}

IMPORTANT:
- If the chunk is relevant, the "answer" field must be SHORT and answer-only.
- Do not use a full sentence in the "answer" field."""

MAP_PROMPT = """Chunk:
{chunk}

Question: {question}

If this chunk contains relevant info, answer the question. Otherwise set relevant=false.
IMPORTANT: If relevant=true, the "answer" field must be SHORT and answer-only.
Respond as JSON with keys: answer, confidence, relevant."""

REDUCE_SYSTEM = """You synthesize a final answer from multiple partial answers.
Respond with JSON: {"answer": "...", "confidence": 0.0-1.0, "reasoning": "..."}

IMPORTANT:
- The "answer" field must be SHORT and answer-only.
- Do not use a full sentence in the "answer" field."""

REDUCE_PROMPT = """Question: {question}

Partial answers from different document sections:
{partial_answers}

Synthesize the best final answer from these partial answers.
IMPORTANT: The "answer" field must be SHORT and answer-only.
Respond as JSON with keys: answer, confidence, reasoning."""


class MapReduceController(BaseController):
    """Map-reduce over retrieved chunks."""

    method_name = "mapreduce"

    def __init__(self, model: str | None = None):
        self.model = model or settings.model_fast

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        trace = TraceNode(action="mapreduce", input=question)

        retriever = kwargs.get("retriever")
        if retriever is None:
            retriever = HybridRetriever(store, cache_key=kwargs.get("cache_key", ""))

        search = SearchTool(retriever)
        chunk_ids = search(question, top_k=settings.mapreduce_map_chunks)

        trace.add_child(TraceNode(
            action="search", input=question, output=str(chunk_ids),
        ))

        # Map phase
        map_answers = []
        for cid in chunk_ids:
            chunk = store.get_chunk(cid)
            if chunk is None:
                continue

            prompt = MAP_PROMPT.format(chunk=chunk.text, question=question)
            result = generate_json(prompt, model=self.model, system=MAP_SYSTEM)

            map_node = TraceNode(
                action="map", input=f"chunk_{cid}",
                output=result.get("answer", ""),
                metadata={"confidence": result.get("confidence", 0.0)},
            )
            trace.add_child(map_node)

            if result.get("relevant", False):
                map_answers.append({
                    "chunk_id": cid,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.0),
                })

        if not map_answers:
            trace.output = "No relevant information found."
            return ControllerResult(
                answer="No relevant information found.",
                confidence=0.0,
                method=self.method_name,
                trace=trace,
            )

        # Reduce phase
        partial_text = "\n".join(
            f"  - [Chunk {a['chunk_id']}, conf={a['confidence']:.2f}]: {a['answer']}"
            for a in map_answers
        )
        prompt = REDUCE_PROMPT.format(question=question, partial_answers=partial_text)
        result = generate_json(prompt, model=self.model, system=REDUCE_SYSTEM)

        trace.add_child(TraceNode(
            action="reduce", input=f"{len(map_answers)} partial answers",
            output=result.get("answer", ""),
            metadata={"confidence": result.get("confidence", 0.0)},
        ))
        trace.output = result.get("answer", "")
        trace.metadata["confidence"] = result.get("confidence", 0.0)

        return ControllerResult(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            method=self.method_name,
            trace=trace,
            metadata={"reasoning": result.get("reasoning", "")},
        )
