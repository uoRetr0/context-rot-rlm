"""Tests for RLM controller recursion behavior."""

from src.controllers.rlm_controller import RLMController
from src.tools.reason_tool import ReasonResult
from src.trace.tracer import TraceNode


class _FakeStore:
    num_chunks = 200


class _FakeRead:
    def __call__(self, chunk_ids: list[int]) -> str:
        return "\n".join(f"[Chunk {cid}] evidence_{cid}" for cid in chunk_ids)


class _FakeSearch:
    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []
        self.results = {
            "root": [0],
            "seed terms": [99],
            "sub question": [1, 2],
        }

    def __call__(self, query: str, top_k: int | None = None) -> list[int]:
        self.calls.append((query, top_k))
        return self.results.get(query, [])


class _FakeReason:
    def reason(self, question: str, evidence: str) -> ReasonResult:
        if question == "root":
            return ReasonResult(
                answer="partial",
                confidence=0.2,
                reasoning="missing bridge fact",
                evidence_used=[0],
                requires_multi_hop=True,
            )

        answer = "seeded answer" if "evidence_99" in evidence else "missed answer"
        return ReasonResult(
            answer=answer,
            confidence=0.9,
            reasoning="used seeded evidence" if "evidence_99" in evidence else "missed seed",
            evidence_used=[99] if "evidence_99" in evidence else [1],
            requires_multi_hop=False,
        )

    def decompose(
        self,
        question: str,
        partial_answer: str,
        confidence: float,
        evidence_summary: str = "",
    ) -> tuple[list[str], list[str]]:
        return ["sub question"], ["seed terms"]

    def merge(
        self,
        question: str,
        initial_answer: str,
        sub_answers: list[dict[str, object]],
        initial_confidence: float = 0.0,
        initial_reasoning: str = "",
    ) -> ReasonResult:
        best = sub_answers[0]
        return ReasonResult(
            answer=str(best["answer"]),
            confidence=float(best["confidence"]),
            reasoning="merged",
            evidence_used=[],
            requires_multi_hop=False,
        )


def test_recursive_answer_uses_seeded_chunks(monkeypatch):
    controller = RLMController(max_depth=1, confidence_threshold=0.5, max_sub_questions=1)
    search = _FakeSearch()
    read = _FakeRead()
    reason = _FakeReason()
    trace = TraceNode(action="rlm", input="root")

    monkeypatch.setattr("src.controllers.rlm_controller.tracker.check_budget", lambda: None)

    result = controller._recursive_answer(
        question="root",
        search=search,
        read=read,
        reason=reason,
        store=_FakeStore(),
        depth=0,
        trace=trace,
        accumulated_chunk_ids=set(),
    )

    sub_nodes = [child for child in trace.children if child.action == "sub_rlm"]
    assert result.answer == "seeded answer"
    assert sub_nodes
    assert any(child.action == "seed_search" for child in sub_nodes[0].children)
