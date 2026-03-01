"""REASON tool: question + evidence → answer with confidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.gemini_client import generate_json

logger = logging.getLogger(__name__)

REASON_SYSTEM = """You are a precise question-answering system. Given a question and evidence passages, provide an answer based ONLY on the evidence.

Respond with a JSON object containing:
- "answer": your answer (string, concise)
- "confidence": how confident you are (float, 0.0 to 1.0)
- "reasoning": brief chain of thought (string)
- "evidence_used": list of chunk IDs you relied on (list of ints)"""

REASON_PROMPT = """Question: {question}

Evidence:
{evidence}

Provide your answer as JSON with keys: answer, confidence, reasoning, evidence_used."""

DECOMPOSE_SYSTEM = """You decompose complex questions into simpler sub-questions.
Respond with a JSON object containing:
- "sub_questions": list of 2-3 simpler questions that together help answer the original question"""

DECOMPOSE_PROMPT = """Original question: {question}

Current partial answer: {partial_answer}

The current answer has low confidence ({confidence:.2f}). Decompose the original question into 2-3 simpler sub-questions that would help gather more specific evidence.

Respond as JSON with key "sub_questions" (list of strings)."""

MERGE_SYSTEM = """You synthesize a final answer from sub-answers.
Respond with a JSON object containing:
- "answer": final synthesized answer (string)
- "confidence": confidence in final answer (float, 0.0 to 1.0)
- "reasoning": how you combined the sub-answers (string)"""

MERGE_PROMPT = """Original question: {question}

Initial answer: {initial_answer}

Sub-question answers:
{sub_answers}

Synthesize a final, accurate answer from all the information above.
Respond as JSON with keys: answer, confidence, reasoning."""


@dataclass
class ReasonResult:
    answer: str
    confidence: float
    reasoning: str
    evidence_used: list[int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasonResult:
        return cls(
            answer=str(d.get("answer", "")),
            confidence=float(d.get("confidence", 0.0)),
            reasoning=str(d.get("reasoning", "")),
            evidence_used=d.get("evidence_used", []),
        )


class ReasonTool:
    """Answers questions given evidence, with confidence scoring."""

    def __init__(self, model: str | None = None):
        self.model = model

    def reason(self, question: str, evidence: str) -> ReasonResult:
        """Answer a question given evidence text."""
        prompt = REASON_PROMPT.format(question=question, evidence=evidence)
        result = generate_json(
            prompt, model=self.model, system=REASON_SYSTEM
        )
        return ReasonResult.from_dict(result)

    def decompose(
        self, question: str, partial_answer: str, confidence: float
    ) -> list[str]:
        """Decompose a question into sub-questions."""
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            partial_answer=partial_answer,
            confidence=confidence,
        )
        result = generate_json(
            prompt, model=self.model, system=DECOMPOSE_SYSTEM
        )
        subs = result.get("sub_questions", [])
        return [str(s) for s in subs]

    def merge(
        self,
        question: str,
        initial_answer: str,
        sub_answers: list[dict[str, str]],
    ) -> ReasonResult:
        """Merge sub-answers into a final answer."""
        sub_text = "\n".join(
            f"  Q: {sa['question']}\n  A: {sa['answer']}"
            for sa in sub_answers
        )
        prompt = MERGE_PROMPT.format(
            question=question,
            initial_answer=initial_answer,
            sub_answers=sub_text,
        )
        result = generate_json(
            prompt, model=self.model, system=MERGE_SYSTEM
        )
        return ReasonResult.from_dict(result)
