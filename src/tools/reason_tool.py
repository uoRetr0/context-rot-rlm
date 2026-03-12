"""REASON tool: question + evidence → answer with confidence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.gemini_client import generate_json

logger = logging.getLogger(__name__)

REASON_SYSTEM = """You are a precise question-answering system. Given a question and evidence passages, provide an answer based ONLY on the evidence.

CONFIDENCE CALIBRATION (follow this strictly):
- 0.9–1.0: The answer is DIRECTLY and EXPLICITLY stated in the evidence.
- 0.7–0.8: The answer is strongly supported but requires minor inference.
- 0.4–0.6: Partial evidence exists — some relevant facts but gaps remain.
- 0.2–0.3: Weak evidence — mostly guessing based on tangential info.
- 0.0–0.1: No relevant evidence found; answer is a pure guess.

Do NOT inflate confidence. If the evidence does not clearly support the answer, confidence MUST be below 0.5.

MULTI-HOP DETECTION:
A question requires multi-hop reasoning if the QUESTION STRUCTURE requires connecting 2+ separate facts to derive the answer — regardless of whether those facts are visible in the evidence.

Multi-hop examples (requires_multi_hop=true):
- "Where was X's workplace located?" → find X's org, THEN find org's location (2 hops)
- "What is the capital of the country where X was born?" → find birthplace, THEN find capital
- "Where does the organization founded by X's mentor operate?" → find mentor, find org, find location (3 hops)
- Any question where the answer is about an ATTRIBUTE OF AN ENTITY that must first be identified

Single-hop examples (requires_multi_hop=false):
- "What is X?" → directly stated or not
- "When did Y happen?" → one fact lookup
- "Who founded Z?" → one fact lookup

IMPORTANT: If your answer describes an INTERMEDIATE entity (e.g., answering "where is X's workplace" with the workplace NAME rather than its LOCATION), that means multi-hop is needed and you may have only completed the first hop.

Respond with a JSON object containing:
- "answer": the answer ONLY — as short as possible (a name, number, place, or brief phrase). Do NOT include reasoning, context, or full sentences in the answer field.
- "confidence": how confident you are (float, 0.0 to 1.0, calibrated per above)
- "reasoning": brief chain of thought (string)
- "evidence_used": list of chunk IDs you relied on (list of ints)
- "requires_multi_hop": whether the question requires chaining multiple facts (boolean)"""

REASON_PROMPT = """Question: {question}

Evidence:
{evidence}

Provide your answer as JSON with keys: answer, confidence, reasoning, evidence_used, requires_multi_hop.
IMPORTANT: The "answer" field must be SHORT — just the key fact (e.g., "Paris", "42", "Dr. Smith"). Put explanations in "reasoning" instead.
Remember: calibrate confidence strictly. Only 0.9+ if the answer is directly stated."""

DECOMPOSE_SYSTEM = """You decompose complex questions into targeted sub-questions to fill evidence gaps.

Rules for sub-questions:
1. Each sub-question must be SPECIFIC and SEARCHABLE — include entity names, dates, or key terms from the original question.
2. Sub-questions should target GAPS in the current evidence, not repeat what is already known.
3. For multi-hop questions, decompose into a chain: first find entity X, then use X to find Y.
4. Each sub-question should be answerable independently from a document search.

Respond with a JSON object containing:
- "sub_questions": list of 2-3 specific sub-questions (for reasoning)
- "search_queries": list of 2-3 optimized search queries (short keyword phrases for retrieval, one per sub-question)"""

DECOMPOSE_PROMPT = """Original question: {question}

Current partial answer: {partial_answer} (confidence: {confidence:.2f})

Evidence gathered so far:
{evidence_summary}

The current answer has LOW confidence. Identify what specific information is MISSING and generate sub-questions to find it.

For multi-hop questions (e.g., "What is the capital of the country where X was born?"):
- First sub-question: find the intermediate fact (e.g., "Where was X born?")
- Second sub-question: use that to find the final answer (e.g., "What is the capital of [country]?")

Respond as JSON with keys "sub_questions" (list of strings) and "search_queries" (list of short keyword strings for search)."""

MERGE_SYSTEM = """You synthesize a final answer from an initial answer and sub-question answers.

Rules:
1. Prefer sub-answers with HIGH confidence over those with low confidence.
2. Chain facts across sub-answers for multi-hop reasoning (A→B→C).
3. If sub-answers contradict each other, go with the one that has stronger evidence.
4. Your final confidence should reflect the weakest link in the reasoning chain.
5. The answer must be SHORT — just the final fact, name, number, or brief phrase. No full sentences.

Respond with a JSON object containing:
- "answer": final answer ONLY — as short as possible (e.g., "Paris", "42", "Dr. Smith"). No sentences or explanations.
- "confidence": confidence in final answer (float, 0.0 to 1.0)
- "reasoning": how you combined the sub-answers (string)"""

MERGE_PROMPT = """Original question: {question}

Initial answer: {initial_answer} (confidence: {initial_confidence:.2f})
Initial reasoning: {initial_reasoning}

Sub-question answers:
{sub_answers}

Synthesize a final, accurate answer. Chain facts from sub-answers if this is a multi-hop question.
IMPORTANT: The "answer" must be SHORT — just the key fact (name, place, number). Put your reasoning in the "reasoning" field.
Respond as JSON with keys: answer, confidence, reasoning."""


@dataclass
class ReasonResult:
    answer: str
    confidence: float
    reasoning: str
    evidence_used: list[int]
    requires_multi_hop: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasonResult:
        raw_conf = float(d.get("confidence", 0.0))
        clamped = max(0.0, min(1.0, raw_conf))
        return cls(
            answer=str(d.get("answer", "")),
            confidence=clamped,
            reasoning=str(d.get("reasoning", "")),
            evidence_used=d.get("evidence_used", []),
            requires_multi_hop=bool(d.get("requires_multi_hop", False)),
        )


def _answer_grounded(answer: str, evidence: str) -> bool:
    """Check if the answer (or key tokens) appear in the evidence text."""
    evidence_lower = evidence.lower()
    answer_lower = answer.lower().strip()

    # Direct substring match
    if answer_lower in evidence_lower:
        return True

    # Check if majority of answer tokens appear in evidence
    answer_tokens = set(answer_lower.split())
    # Remove common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                 "and", "or", "for", "on", "at", "by", "it", "that", "this", "with"}
    content_tokens = answer_tokens - stopwords
    if not content_tokens:
        return True  # All stopwords, can't judge

    evidence_tokens = set(evidence_lower.split())
    overlap = content_tokens & evidence_tokens
    return len(overlap) / len(content_tokens) >= 0.5


def _calibrate_confidence(result: ReasonResult, evidence: str) -> ReasonResult:
    """Heuristic post-processing to counter Gemini's confidence inflation.

    Strategy:
    - Single-hop + grounded: mild deflation (preserve good answers, no recursion)
    - Multi-hop + grounded: moderate deflation (trigger recursion for fact-chaining)
    - Not grounded: hard deflation (force recursion)
    """
    conf = result.confidence
    grounded = _answer_grounded(result.answer, evidence)
    evidence_words = len(evidence.split())

    if not grounded:
        # Answer NOT grounded — likely confabulating.
        # Force recursion by capping well below threshold.
        conf = min(conf * 0.4, 0.35)
    elif result.requires_multi_hop:
        # Multi-hop question with grounded answer — the answer tokens appear
        # in evidence but may not be correctly chained. Deflate to trigger
        # decomposition so sub-questions can verify each hop independently.
        # 0.9 * 0.50 = 0.45 → below 0.5 threshold, triggers recursion.
        conf *= 0.50

        # Thin evidence penalty
        if evidence_words < 50:
            conf *= 0.5
        elif evidence_words < 150:
            conf *= 0.7
    else:
        # Single-hop + grounded — mild deflation only.
        # 0.9 * 0.85 = 0.765 → stays above 0.5 threshold, no recursion.
        conf *= 0.85

        # Still penalize thin evidence even if grounded
        if evidence_words < 50:
            conf *= 0.5
        elif evidence_words < 150:
            conf *= 0.7

    # Penalize very short answers (likely guesses)
    answer_words = len(result.answer.split())
    if answer_words < 3:
        conf *= 0.8

    # Few evidence chunks used → less certain
    if len(result.evidence_used) <= 1:
        conf *= 0.85

    result.confidence = max(0.0, min(1.0, conf))
    return result


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
        rr = ReasonResult.from_dict(result)
        return _calibrate_confidence(rr, evidence)

    def decompose(
        self,
        question: str,
        partial_answer: str,
        confidence: float,
        evidence_summary: str = "",
    ) -> tuple[list[str], list[str]]:
        """Decompose a question into sub-questions and search queries."""
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            partial_answer=partial_answer,
            confidence=confidence,
            evidence_summary=evidence_summary or "(no evidence yet)",
        )
        result = generate_json(
            prompt, model=self.model, system=DECOMPOSE_SYSTEM
        )
        subs = result.get("sub_questions", [])
        queries = result.get("search_queries", subs)  # fallback to sub_questions
        return [str(s) for s in subs], [str(q) for q in queries]

    def merge(
        self,
        question: str,
        initial_answer: str,
        sub_answers: list[dict[str, Any]],
        initial_confidence: float = 0.0,
        initial_reasoning: str = "",
    ) -> ReasonResult:
        """Merge sub-answers into a final answer."""
        sub_text = "\n".join(
            f"  Q: {sa['question']}\n"
            f"  A: {sa['answer']} (confidence: {sa.get('confidence', '?')})\n"
            f"  Reasoning: {sa.get('reasoning', 'N/A')}\n"
            f"  Evidence chunks: {sa.get('evidence_used', 'N/A')}"
            for sa in sub_answers
        )
        prompt = MERGE_PROMPT.format(
            question=question,
            initial_answer=initial_answer,
            initial_confidence=initial_confidence,
            initial_reasoning=initial_reasoning or "N/A",
            sub_answers=sub_text,
        )
        result = generate_json(
            prompt, model=self.model, system=MERGE_SYSTEM
        )
        return ReasonResult.from_dict(result)
