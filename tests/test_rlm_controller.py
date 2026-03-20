"""Tests for the REPL-style RLM controller."""

from src.config import settings
from src.controllers.rlm_controller import RLMController
from src.environment.document_store import DocumentStore


class _FakeRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        results = self.responses.get(query, [])
        return results[: top_k or len(results)]


def test_rlm_controller_executes_repl_search_and_finish(monkeypatch):
    calls = iter([
        {
            "answer": "beta",
            "confidence": 0.8,
            "reasoning": "The bootstrap snippet already contains the answer.",
        },
        {
            "thought": "Inspect the top chunk.",
            "code": "hits = search('alpha', top_k=1)\ntext = read_chunks(hits)\nprint(text)",
        },
        {
            "thought": "Return the answer.",
            "code": "finish('beta', 0.9, 'Found beta in the retrieved chunk.')",
        },
        {
            "answer": "beta",
            "confidence": 0.95,
            "reasoning": "The gathered evidence directly supports beta.",
        },
    ])
    monkeypatch.setattr(
        "src.controllers.rlm_controller.generate_json",
        lambda *args, **kwargs: next(calls),
    )

    store = DocumentStore()
    store.ingest("alpha beta gamma delta", doc_id="doc")
    controller = RLMController(max_depth=2)

    result = controller.answer(
        "What comes after alpha?",
        store,
        retriever=_FakeRetriever({"alpha": [(0, 1.0)]}),
    )

    assert result.answer == "beta"
    assert result.confidence == 0.95
    assert result.metadata["controller_style"] == "repl_recursive"
    assert result.trace is not None
    assert any(child.action == "plan" for child in result.trace.children)
    assert any(child.action == "verify_final" for child in result.trace.children)


def test_rlm_controller_supports_recursive_llm_query(monkeypatch):
    calls = iter([
        {
            "answer": "beta",
            "confidence": 0.7,
            "reasoning": "Bootstrap snippet suggests beta.",
        },
        {
            "thought": "Read the candidate chunk and recurse on it.",
            "code": (
                "hits = search('alpha', top_k=1)\n"
                "text = read_chunks(hits)\n"
                "sub = llm_query('What is the key answer?', context=text)\n"
                "finish(sub['answer'], sub['confidence'], sub['reasoning'])"
            ),
        },
        {
            "answer": "beta",
            "confidence": 0.9,
            "reasoning": "Sub-context snippet contains beta.",
        },
        {
            "thought": "Answer inside the smaller context.",
            "code": "finish('beta', 0.8, 'Solved inside the sub-context.')",
        },
        {
            "answer": "beta",
            "confidence": 0.85,
            "reasoning": "The original question is directly answered by beta.",
        },
    ])
    monkeypatch.setattr(
        "src.controllers.rlm_controller.generate_json",
        lambda *args, **kwargs: next(calls),
    )

    store = DocumentStore()
    store.ingest("alpha beta gamma delta", doc_id="doc")
    controller = RLMController(max_depth=2)

    result = controller.answer(
        "What is the key answer?",
        store,
        retriever=_FakeRetriever({"alpha": [(0, 1.0)]}),
    )

    assert result.answer == "beta"
    assert result.trace is not None

    def _has_action(node, action):
        if node.action == action:
            return True
        return any(_has_action(child, action) for child in node.children)

    assert _has_action(result.trace, "llm_query")
    assert _has_action(result.trace, "verify_final")


def test_rlm_controller_bootstrap_uses_configured_initial_chunks(monkeypatch):
    calls = iter([
        {
            "answer": "beta",
            "confidence": 0.7,
            "reasoning": "Bootstrap snippet suggests beta.",
        },
        {
            "thought": "Return the answer immediately.",
            "code": "finish('beta', 0.8, 'Directly supported by the bootstrap evidence.')",
        },
        {
            "answer": "beta",
            "confidence": 0.9,
            "reasoning": "The original question is directly answered by beta.",
        },
    ])
    monkeypatch.setattr(
        "src.controllers.rlm_controller.generate_json",
        lambda *args, **kwargs: next(calls),
    )

    store = DocumentStore()
    store.ingest("alpha beta gamma delta", doc_id="doc")
    retriever = _FakeRetriever({"What comes after alpha?": [(0, 1.0)]})
    controller = RLMController(max_depth=2)

    result = controller.answer(
        "What comes after alpha?",
        store,
        retriever=retriever,
    )

    assert result.answer == "beta"
    assert retriever.calls[0] == ("What comes after alpha?", settings.rlm_initial_chunks)
    assert result.trace.children[0].metadata["top_k"] == settings.rlm_initial_chunks


def test_rlm_controller_verifies_ambiguous_final_answer(monkeypatch):
    calls = iter([
        {
            "answer": "beta",
            "confidence": 0.6,
            "reasoning": "The snippet directly contains beta.",
        },
        {
            "thought": "Finish with the ambiguous summary.",
            "code": "finish('Conflicting information', 0.7, 'There appear to be multiple possibilities.')",
        },
        {
            "answer": "beta",
            "confidence": 0.92,
            "reasoning": "The exact answer to the original question is beta.",
        },
    ])
    monkeypatch.setattr(
        "src.controllers.rlm_controller.generate_json",
        lambda *args, **kwargs: next(calls),
    )

    store = DocumentStore()
    store.ingest("alpha beta gamma delta", doc_id="doc")
    controller = RLMController(max_depth=2)

    result = controller.answer(
        "What comes after alpha?",
        store,
        retriever=_FakeRetriever({"What comes after alpha?": [(0, 1.0)]}),
    )

    assert result.answer == "beta"
    assert result.confidence == 0.92
